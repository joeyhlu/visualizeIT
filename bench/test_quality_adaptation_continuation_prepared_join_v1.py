"""MODEL-FREE SOFTWARE FIXTURES for the explicit continuation consumer join.

No model, checkpoint, dataset asset or reference annotation is opened. Identity
poses, constant RGB/depth/confidence and zero flow are analytical software data.
The prepared view is authored as synthetic fixture bytes; this suite does not
claim physical hardlink transport (the separate Linux core suite covers that).
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack, nullcontext, redirect_stdout
import copy
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_portable_prepare as p
from bench import quality_adaptation_cached_capture_v1 as cached
from bench import quality_adaptation_sharded_v1 as shards
from bench.quality_adaptation_data import HOOKS, crop_rgb_hash, load_npz, CROP_KEYS
from bench.quality_adaptation_prepare import crops_from_retained_capture
from bench.quality_adaptation_refinement_states import state_rows
from bench.quality_ranker_synthetic import NORMALIZED_VIEW_SCHEDULE, validate_normalized_views
from bench.quality_ranker_data import read_split_manifest
from bench.test_quality_adaptation_portable_prepare import known_role_fixture
from bench.test_kaggle_sharded_prepared_v1 import model_free_ui_and_evidence
from scripts import kaggle_prepared_v2 as prepared
from scripts import kaggle_sharded_prepared_v1 as child

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / '.cache/cloud-prepare/test-tmp'
ARCHIVE = ROOT / '.cache/capture-resume-v1/frozen-originals'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw.encoded(value))
    return p.pin(path)


def context(document):
    data = raw.encoded(document)
    return raw.PortableContext(data, hashlib.sha256(data).hexdigest())


class SoftwareJoin:
    def __init__(self, base, *, aggregate_only=False):
        self.base = Path(base)
        self.aggregate_only = aggregate_only
        self.extra = {}
        self.recipe, self.audit = known_role_fixture(self.base)
        self.smoke = self.freeze('smoke0')
        self.group = self.freeze('group1')
        self.legacy_seal(self.smoke)
        if aggregate_only:
            for sid in ('group2', 'group3'):
                self.extra[sid] = self.freeze(sid)
                self.legacy_seal(self.extra[sid])
        # A software partition uses the genuine freeze/projection algorithm.
        smoke_migration = self.migration()
        with p.source_migration_scope(smoke_migration):
            frozen = shards.freeze_partition(self.recipe, self.audit,
                dict(context=self.smoke['old_pin'], raw_seal=p.pin(self.smoke['root'] / 'raw-seal.json')),
                self.base / 'partition')
        self.partition = frozen['partition']
        self.smoke_projection = frozen['imported_shard0_projection']
        self.make_terminal()
        self.capture()
        self.make_view()
        self.migration_pin = self.migration()
        self.projection = shards.seal_shard(self.partition, 'group1', self.group['new_pin'], self.view_pin,
            self.base / 'continuation-projection.json', raw_continuation=self.descriptor,
            source_migration=self.migration_pin)

    def freeze(self, sid):
        directory = self.base / sid; directory.mkdir()
        evidence = write(directory / 'software-parity.json', dict(MODEL_FREE_SOFTWARE_FIXTURE=True))
        root = directory / 'raw-run'
        doc = raw.context_document(output_root=root, scratch_root=directory / 'scratch', device='cpu',
            parity=dict(reference=evidence, candidate=evidence))
        current = context(doc)
        recipe = shards._projected_recipe(self.recipe, shards.TRAIN_GROUPS[sid])
        recipe_pin = write(directory / 'source-recipe.json', recipe)
        raw.freeze(recipe_pin['path'], recipe_pin['sha256'], current)
        old_doc = copy.deepcopy(doc)
        old_doc['source_files']['bench/quality_adaptation_portable.py'] = p.sha256(ARCHIVE / 'bench/quality_adaptation_portable.py')
        previous = context(old_doc)
        (root / 'context.json').write_bytes(previous.raw)
        freeze = raw.read_json(root / 'freeze.json'); freeze.update(context_sha256=previous.sha256, elapsed_seconds=1.)
        write(root / 'freeze.json', freeze)
        physical = raw.physical_rows(recipe, raw.prospective(recipe))
        for frame in raw._active(physical):
            for kind in ('render', 'bank', 'mask', 'seed'):
                frame[kind] = dict(status='unavailable', failure_reason='SOFTWARE_FIXTURE_OBSERVED_FAILURE')
        states = state_rows(dict(rows=physical))
        for row in states:
            row['failure_reason'] = ('SOFTWARE_FIXTURE_OBSERVED_FAILURE'
                if row['split'] in ('train', 'validation') else 'sealed_metadata_only')
        write(root / 'physical-index.json', dict(rows=physical))
        write(root / 'input-plan.json', raw.capture_plan(previous, physical))
        write(root / 'raw/capture-index.json', dict(rows=states, training_ready=False))
        for stage in ('render', 'banks', 'masks'):
            write(root / ('stage-' + stage + '.json'), dict(stage=stage, context_sha256=previous.sha256,
                status='completed', error=None, elapsed_seconds=1., test_payload_reads=0, training_ready=False))
        return dict(root=root, recipe=recipe, old=previous, new=current, physical=physical, states=states,
            old_pin=p.pin(root / 'context.json'), new_pin=write(directory / 'successor-context.json', doc),
            audit=write(directory / 'audit.json', shards._projected_audit(self.audit, recipe)))

    def migration(self):
        names = ('bench/quality_adaptation_portable.py', 'bench/quality_adaptation_portable_prepare.py',
                 'scripts/kaggle_prepared_v2.py', 'bench/quality_adaptation_sharded_v1.py')
        sources = {}
        for name in names:
            archive = p.pin(ARCHIVE / name)
            sources[name] = dict(original=dict(archive, path=(ROOT / name).as_posix()),
                archive=archive, successor=p.pin(ROOT / name))
        return write(self.base / 'migration.json', dict(schema='visualizeit_prepared_source_migration_v1',
            sources=sources, contexts={value['old'].sha256: value['new_pin']
                for value in (self.smoke, self.group, *self.extra.values())}))

    def legacy_seal(self, value):
        root = value['root']; old = value['old']; states = value['states']
        write(root / 'raw/capture-receipt.json', dict(schema='kaggle_raw_capture_receipt_v1',
            context_sha256=old.sha256, input_plan_sha256=p.sha256(root / 'input-plan.json'),
            fatal_error=None, cleanup_errors=[], physical_frames_requested=len(value['physical']),
            refinement_states_requested=len(states), annotation_inputs=[], test_payload_reads=0, training_ready=False))
        receipt = dict(stage='capture', context_sha256=old.sha256, status='completed', error=None,
            elapsed_seconds=1., test_payload_reads=0, training_ready=False)
        write(root / 'stage-capture.json', receipt)
        with raw._successor_sources(old, value['new'], root / 'raw'), patch.object(raw, '_require_stage'):
            raw.seal_capture(old)
        write(root / 'stage-seal-capture.json', dict(receipt, stage='seal-capture'))

    def materialize(self, output, frame, states, level):
        k = np.array([[280., 0., 139.5], [0., 280., 139.5], [0., 0., 1.]])
        # Analytical identity projection: native and crop cameras coincide,
        # object depth is exactly 1000 mm, flow is zero, confidence is .5.
        retained = dict(query_rgb_crop=np.full((1, 3, 280, 280), level, np.float32),
            template_rgb=np.full((1, 3, 280, 280), .1, np.float32),
            rendered_depth_mm=np.full((280, 280), 1000., np.float32), rendered_mask=np.ones((280, 280), bool),
            crop_k=k, current_crop_camera_from_object_mm=np.eye(4))
        crops = crops_from_retained_capture(retained, preserve_float_rgb=True)
        geometry = dict(crop_from_orig=np.eye(4), crop_k=k, native_k=k,
            current_crop_camera_from_object_mm=np.eye(4), rendered_depth_mm=retained['rendered_depth_mm'],
            rendered_mask=retained['rendered_mask'], actual_flow=np.zeros((280, 280, 2), np.float32),
            actual_confidence=np.full((280, 280), .5, np.float32))
        frame['seed'] = dict(status='available', poses=[np.eye(4).tolist()], selection='SOFTWARE_FIXTURE')
        row = next(r for r in states if r['id'].startswith(frame['id'] + ':iteration'))
        ordinal = states.index(row); descriptors = {}
        for key, name, arrays in (('crops', 'crop', crops), ('capture', 'capture', geometry)):
            path = output / f'{name}-{ordinal:02d}.npz'; np.savez(path, **arrays)
            descriptors[key] = dict(p.pin(path), path=path.name)
        row.pop('failure_reason', None); row.pop('unavailable_provenance', None)
        row.update(status='available', **descriptors, query_rgb_sha256=crop_rgb_hash(crops['query_rgb']),
            pnp_state='SOFTWARE_FIXTURE_TERMINAL', native_calibration=k.tolist(),
            sampling_map=dict(integer_centers=280**2, max_native_pixel_difference=0.))

    def make_terminal(self):
        value = self.group; root = value['root']; physical, states = value['physical'], value['states']
        self.old_adapter = p.pin(ARCHIVE / 'bench/quality_adaptation_cached_capture_v1.py')
        write(root / cached.POLICY_FILE, dict(adapter=dict(self.old_adapter,
            path=(ROOT / 'bench/quality_adaptation_cached_capture_v1.py').as_posix()),
            MODEL_FREE_SOFTWARE_FIXTURE=True))
        for split, level in (('validation', .25), ('train', .5)):
            frame = next(r for r in physical if r['split'] == split)
            self.materialize(root / 'raw', frame, states, level)
        self.pending = raw._active(physical)[-1]
        # Software-only observed-input and generation namespaces are distinct.
        # No real mesh/video payload exists; renderer/media admission is the
        # explicit external backend double in this test.
        self.vertices = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        geometry = validate_normalized_views(self.vertices)
        split = read_split_manifest(root / 'split.json')
        selected = [next(r for r in physical if r['split'] == split_name) for split_name in ('validation', 'train')]
        for ordinal, frame in enumerate([*selected, self.pending]):
            if self.aggregate_only:
                # Genuine small physical files retained at the predecessor
                # root, authenticated before capture and hardlinked by the real
                # aggregator. No mesh/media/renderer asset is represented.
                bank = root / 'banks' / str(ordinal)
                frame['bank'] = dict(status='available',
                    bank=write(bank / 'bank.json', dict(SOFTWARE_FIXTURE_BANK=ordinal)),
                    cnos_bank=write(bank / 'cnos.json', dict(SOFTWARE_FIXTURE_CNOS=ordinal)))
                continue
            view = root / 'views' / str(ordinal); bundle = view / 'input'
            manifest = write(bundle / 'input.json', dict(asset='object.glb', object_id=frame['object_id'],
                physical_object_id=frame['physical_object_id'], MODEL_FREE_SOFTWARE_FIXTURE=True))
            video = bundle / 'video.mp4'; video.write_bytes(b'SOFTWARE FIXTURE: NEVER DECODED')
            frame['render'] = dict(status='available', input=manifest, video=p.pin(video),
                mesh=dict(frame['mesh_recipe'], path=(bundle / 'MUST_NOT_OPEN-object.glb').as_posix()))
            write(view / 'generation.json', dict(kind='quality-ranker-synthetic-generation-v1',
                view_mode='normalized24-v1', view_index=frame['view_index'],
                view_schedule=[list(r) for r in NORMALIZED_VIEW_SCHEDULE], sequence_id=frame['sequence_id'],
                split=frame['split'], physical_object_id=frame['physical_object_id'],
                mesh_sha256=frame['mesh_recipe']['sha256'], video_sha256=frame['render']['video']['sha256'],
                input_sha256=manifest['sha256'], split_file_sha256=p.sha256(root / 'split.json'),
                split_manifest_sha256=split['manifest_sha256'], source_sha256=p.sha256(ROOT / 'bench/quality_ranker_synthetic.py'),
                center_m=geometry['center_m'].tolist(), radius_m=geometry['radius_m'], distance_m=geometry['distance_m']))
        self.pending['seed'] = {'status': 'pending'}
        pending_state = next(r for r in states if r['id'].startswith(self.pending['id'] + ':iteration'))
        pending_state.clear(); pending_state.update(next(r for r in state_rows(dict(rows=physical)) if r['id'].startswith(self.pending['id'] + ':iteration')))
        write(root / 'physical-index.json', dict(rows=physical))
        write(root / 'input-plan.json', raw.capture_plan(value['old'], physical))
        write(root / 'raw/capture-index.json', dict(rows=states, training_ready=False))
        pins = {path.name: p.pin(path) for path in root.glob('*.json')}
        pins.update({'raw/capture-index.json': p.pin(root / 'raw/capture-index.json'),
            'raw/capture-receipt.json': None, 'stage-capture.json': None})
        outcomes = []
        for frame in physical:
            status = ('sealed' if frame['split'] not in ('train', 'validation') else 'unattempted'
                if frame is self.pending else 'completed' if frame['seed']['status'] == 'available' else 'failed')
            outcomes.append(dict(id=frame['id'], outcome=status))
        self.terminal = write(self.base / 'terminal.json', dict(schema='visualizeit_raw_capture_terminal_resume_v1',
            context_sha256=value['old'].sha256, metadata=pins,
            source_archive={'bench/quality_adaptation_portable.py': p.pin(ARCHIVE / 'bench/quality_adaptation_portable.py')},
            job=dict(pid=5611, returncode=1, process_start_token='SOFTWARE_FIXTURE_TOKEN',
                process_start_token_at_poll='SOFTWARE_FIXTURE_TOKEN', terminal_observed_unix_s=1000.,
                absence_observed_unix_s=1001., root_and_descendants_absent=True, gpu_compute_pids=[], gpu_used_bytes=0),
            units=outcomes, capture_elapsed_seconds=1., prior_history=None))

    def capture(self):
        value = self.group
        @contextmanager
        def guard(old, new, output, seconds, *, cloud_execution):
            with raw._successor_sources(old, new, output): yield
        @contextmanager
        def backend(old, data, digest, inputs, states, output):
            yield lambda frame: self.materialize(output, frame, states, .75)
        self.allowance = write(self.base / 'allowance.json', dict(
            schema='visualizeit_raw_capture_continuation_allowance_v1', operation='capture_continuation',
            predecessor_context_sha256=value['old'].sha256, successor_context_sha256=value['new'].sha256,
            terminal=self.terminal, additional_allowance=True, max_seconds=300., attempt_id='software-fixture'))
        with patch.object(raw, '_capture_continuation_guard', guard), patch.object(raw, '_capture_backend', backend):
            raw.continue_capture(value['old'], value['new'], self.terminal, self.allowance, cloud_execution=True)
        self.attempt = value['root'] / 'capture-continuations/software-fixture'
        self.receipt = p.pin(self.attempt / 'receipt.json')

    def make_view(self):
        # Independent authored software view; no production hardlink fallback.
        value = self.group; root = value['root']; view = self.attempt / 'preparation-view'
        (view / 'raw').mkdir(parents=True)
        for name in ('recipe.json', 'scale-plan.json', 'split.json'):
            (view / name).write_bytes((root / name).read_bytes())
        (view / 'successor-source-context.json').write_bytes(value['new'].raw)
        receipt = p.metadata(self.receipt); physical = p.metadata(receipt['physical_index'])['rows']
        captures = p.metadata(receipt['capture_index'])['rows']
        for row in captures:
            if row['status'] != 'available': continue
            origin = Path(receipt['artifact_roots'][row['id'].rsplit(':iteration', 1)[0]])
            for key in ('crops', 'capture'):
                (view / 'raw' / row[key]['path']).write_bytes((origin / row[key]['path']).read_bytes())
        artifacts = {path.relative_to(view).as_posix(): p.pin(path) for path in view.rglob('*') if path.is_file()}
        self.adapter = p.pin(Path(cached.__file__))
        self.view_pin = write(view / 'raw-seal.json', p.sealed(dict(
            schema='visualizeit_raw_capture_continuation_seal_v1', continuation_receipt=self.receipt,
            predecessor_terminal=self.terminal, adapter=self.adapter, predecessor_adapter_archive=self.old_adapter,
            predecessor_context_sha256=value['old'].sha256, successor_context_sha256=value['new'].sha256,
            artifact_roots=receipt['artifact_roots'], physical_frames=physical, refinement_states=captures,
            artifacts=artifacts, join_elapsed_seconds=1., cumulative_elapsed_seconds=receipt['cumulative_elapsed_seconds'] + 1.,
            original_cumulative_budget_compliance=False, legacy_raw_seal=False, annotation_inputs=[],
            test_payload_reads=0, training_ready=False)))
        self.descriptor = write(self.base / 'continuation.json', dict(
            schema='visualizeit_prepared_capture_continuation_v1', predecessor_context=value['old_pin'],
            successor_context=value['new_pin'], terminal=self.terminal, receipt=self.receipt,
            adapter=self.adapter, predecessor_adapter_archive=self.old_adapter, view_seal=self.view_pin))


class ContinuationPreparedJoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMP.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=TEMP)
        cls.stack = ExitStack()
        cls.stack.enter_context(patch.object(raw, 'resource_check', return_value={}))
        # External observed-input/model-runtime boundary only. The frozen recipe,
        # source archives, terminal state, indexes and pins are genuinely checked.
        cls.stack.enter_context(patch.object(raw, 'preflight_inputs', return_value={}))
        try:
            cls.f = SoftwareJoin(cls.temporary.name)
        except BaseException:
            cls.stack.close(); cls.temporary.cleanup(); raise

    @classmethod
    def tearDownClass(cls):
        cls.stack.close(); cls.temporary.cleanup()

    def setUp(self):
        self.base = self.f.base / hashlib.sha256(self._testMethodName.encode()).hexdigest()[:8]
        self.base.mkdir()

    def request(self, *, python=None, runtime=None):
        f = self.f; control = self.base / 'control'; control.mkdir()
        fixture = write(self.base / 'software-runtime.json', dict(MODEL_FREE_SOFTWARE_FIXTURE=True))
        sources = {name: p.pin(ROOT / name) for name in prepared.SOURCE_NAMES}
        ui, evidence = model_free_ui_and_evidence(self.base, stage='identity', context=f.group['new_pin'],
            raw_seal=f.view_pin, source_files=sources, storage_scope=[str(self.base)],
            storage_limit=child.AGGREGATE_STORAGE_LIMIT, remaining=3600)
        with patch.object(child, '_check_child_layout'):
            return child.child_request_document(partition=f.partition, shard_id='group1',
                context=f.group['new_pin'], raw_seal=f.view_pin, projection=f.projection,
                exposure_audit=f.group['audit'], label_plan=None, source_root=ROOT,
                source_install=self.base / 'source-install', python=Path(sys.executable).resolve() if python is None else python,
                runtime=fixture if runtime is None else runtime, config=fixture, adapter=ROOT / 'scripts/kaggle_adaptation.py',
                input_root=self.base / 'input', working_root=self.base, execution_root=self.base / 'execution',
                features_root=self.base / 'features', feature_storage_root=self.base / 'feature-parent',
                retained_raw_roots=[str(f.group['root'])], control_root=control, ui_context=ui,
                execution_evidence=evidence, limits=dict(max_seconds=300, max_output_bytes=2 * p.GIB,
                    max_rss_bytes=8 * p.GIB, max_log_bytes=16 * p.MIB, reserve_bytes=p.GIB),
                quota_roots=[str(self.base)], defer_label_plan=True,
                raw_continuation=f.descriptor, source_migration=f.migration_pin)

    def refresh(self, request, stage, remaining):
        ui, evidence = model_free_ui_and_evidence(self.base, stage=stage, context=request['context'],
            raw_seal=request['raw_seal'], source_files=request['source_files'],
            storage_scope=request['quota_roots'], storage_limit=child.AGGREGATE_STORAGE_LIMIT, remaining=remaining)
        return child.child_stage_request(request, stage=stage, ui_context=ui, execution_evidence=evidence)

    def external_boundaries(self):
        stack = ExitStack()
        stack.enter_context(patch.object(child, '_check_child_layout'))
        stack.enter_context(patch.object(p, 'cloud_guard'))
        stack.enter_context(patch.object(child, 'child_quota_check', return_value={'resident_bytes': 0}))
        stack.enter_context(patch.object(p, 'source_runtime', return_value=(
            SimpleNamespace(clean_environment=lambda: {}), ROOT, {'SOFTWARE_FIXTURE': True})))
        stack.enter_context(patch.object(p, 'preparation_evidence', side_effect=lambda request: p.metadata(request['execution_evidence'])))
        stack.enter_context(patch.object(raw, 'verify_neural_environment', return_value={}))
        stack.enter_context(patch.object(p, 'preparation_lock', return_value=nullcontext()))
        return stack

    def fake_external_process(self, request, *, stage, argv, output_root, log_path, **kwargs):
        features = Path(request['features_root'])
        if stage == 'extract':
            self.assertNotIn('label_plan', request)
            self.assertNotIn('child_label_binding', request)
            self.assertFalse((Path(request['control_root']) / 'child-label-plan.json').exists())
            features.mkdir(); manifest = p.metadata(p.pin(self.f.attempt / 'preparation-view/raw/v2-crops.json'))
            lock = p.read_json(ROOT / 'bench/runtime/adaptation-sources.json')
            recipe = dict(schema='frozen_gotrack_decoder_recipe_v1', source_lock_sha256=raw.SOURCE_LOCK_SHA256,
                source_files_sha256=p.canonical(lock['gotrack']['files']), upstream_revision=lock['gotrack']['revision'],
                crop_manifest_sha256=p.canonical(manifest), hooks=list(HOOKS), annotation_inputs=[],
                normalization='upstream_DinoFeatureExtractor_RGB_0_1', rgb_encoding='uint8_div255_or_exact_float32_0_1_v1',
                producer_sha256=lock['gotrack']['files']['bench/quality_adaptation_prepare.py'], device='cpu')
            write(features / 'recipe.json', recipe); rows = []
            for ordinal, row in enumerate(manifest['rows']):
                crops = load_npz(self.f.attempt / 'preparation-view/raw' / row['crops']['path'], CROP_KEYS)
                values = {f'tokens_{hook}': np.full((400, 768), ordinal + 1, np.float32) for hook in HOOKS}
                values.update({key: crops[key] for key in ('template_mask', 'template_xyz_m', 'query_K')})
                values.update(baseline_flow=np.zeros((2, 280, 280), np.float32),
                    baseline_confidence=np.full((280, 280), .5, np.float32))
                path = features / f'feature-{ordinal}.npz'; np.savez(path, **values)
                rows.append(dict(id=row['id'], status='available', crops=row['crops'],
                    query_rgb_sha256=crop_rgb_hash(crops['query_rgb']), features=dict(p.pin(path), path=path.name)))
            write(features / 'index.json', dict(recipe_sha256=p.canonical(recipe), status='completed',
                elapsed_seconds=.1, rows=rows))
        elif stage == 'labels':
            request_path = Path(request['control_root']) / 'child-label-request.json'
            expected = p.sha256(request_path)
            serialized = p.load_request(request_path, expected)
            self.assertEqual(serialized['raw_continuation'], self.f.descriptor)
            self.assertEqual(serialized['source_migration'], self.f.migration_pin)
            with patch.dict(os.environ, VISUALIZEIT_SHARDED_CHILD_LABEL=expected), redirect_stdout(io.StringIO()):
                self.assertEqual(child.main(['child-label-worker', '--request', str(request_path), '--request-sha256', expected]), 0)
        elif stage == 'assemble':
            from bench.quality_adaptation_manifest import assemble
            assemble(features, features / 'labels.json', features / 'manifest.json')
        else:
            raise AssertionError('unexpected external process: ' + stage)
        return dict(status='completed', returncode=0), {'MODEL_FREE_EXTERNAL_PROCESS_FIXTURE': True}

    def test_existing_child_identity_extract_numerical_seal_and_serialized_label_worker(self):
        from bench import quality_adaptation_larger as larger
        from bench import quality_gotrack, quality_runner, quality_adaptation_smoke
        class Renderer:
            vertices_m = self.f.vertices
            def __init__(self, *args, **kwargs): pass
            def close(self): pass
        request = self.request(); identity = request['stage_request_identity_sha256']
        with self.external_boundaries(), patch.object(child, '_run_child_original', side_effect=self.fake_external_process), \
                patch.object(larger, 'configure_torch', return_value=object()), patch.object(larger, 'cleanup_device'), \
                patch.object(quality_gotrack, 'TexturedRenderer', Renderer), \
                patch.object(quality_runner, 'read_input', side_effect=lambda bundle: p.read_json(bundle / 'input.json')), \
                patch.object(quality_adaptation_smoke, 'render_query_labels', return_value=(
                    np.eye(4), np.ones((280, 280), np.float32), np.ones((280, 280), bool))):
            with redirect_stdout(io.StringIO()):
                path = self.base / 'identity-request.json'; pin = write(path, request)
                self.assertEqual(child.main(['child-identity', '--request', str(path), '--request-sha256', pin['sha256']]), 0)
                request = self.refresh(request, 'extract', 3500)
                pin = write(self.base / 'extract-request.json', request)
                self.assertEqual(child.main(['child-extract', '--request', pin['path'], '--request-sha256', pin['sha256']]), 0)
                request = self.refresh(request, 'seal-features', 3400)
                child.child_seal_features(request)
                request = child.bind_child_label_plan(request)
                self.assertEqual(request['stage_request_identity_sha256'], identity)
                request = self.refresh(request, 'labels', 3300)
                child.child_labels(request)
            final = child.finalize_child_request(request)
            joined = child.verify_child_request(final, payloads=False)
        self.assertEqual(joined['raw']['physical'], p.metadata(p.metadata(self.f.receipt)['physical_index'])['rows'])
        self.assertEqual(joined['raw']['captures'], p.metadata(p.metadata(self.f.receipt)['capture_index'])['rows'])
        self.assertEqual(joined['raw']['plan']['global_manifest']['split_manifest'],
            p.read_json(self.f.group['root'] / 'split.json'))
        self.assertEqual(joined['counts']['train']['requested'], 96)
        self.assertEqual(joined['counts']['train']['jointly_usable'], 2)
        self.assertEqual(joined['counts']['validation']['jointly_usable'], 1)
        self.assertEqual(sum(row['status'] == 'available' for row in joined['docs']['labels']['rows']), 3)
        self.assertEqual(final['raw_continuation'], self.f.descriptor)

    def test_mutated_continuation_pins_fail_before_dispatch_or_label_plan_open(self):
        original = p.metadata(self.f.descriptor)
        for field in ('successor_context', 'adapter', 'receipt', 'view_seal'):
            with self.subTest(field=field):
                changed = copy.deepcopy(original); changed[field]['sha256'] = '0' * 64
                descriptor = write(self.base / (field + '.json'), changed)
                request = dict(context=self.f.group['new_pin'], raw_seal=self.f.view_pin,
                    raw_continuation=descriptor, source_migration=self.f.migration_pin,
                    source_root=str(ROOT), source_files={name: p.pin(ROOT / name) for name in prepared.SOURCE_NAMES},
                    exposure_audit=self.f.group['audit'], execution_root=str(self.base / field))
                with patch.object(p, 'execution_identity', side_effect=AssertionError('backend reached')):
                    with self.assertRaises(ValueError): p.prepare_stage('identity', request)

    def test_missing_continuation_descriptor_does_not_admit_new_seal_as_legacy(self):
        with self.assertRaises(ValueError):
            p.prepared_raw_state(self.f.group['new_pin'], self.f.view_pin,
                source_migration=self.f.migration_pin, verify_payloads=False)

    def test_projection_and_request_hash_bind_exact_descriptor(self):
        request = self.request(); original = request['stage_request_identity_sha256']
        changed = copy.deepcopy(request); changed['raw_continuation']['sha256'] = '0' * 64
        self.assertNotEqual(child._child_stage_request_identity(changed), original)
        with self.assertRaisesRegex(ValueError, 'continuation descriptor differs'):
            shards.verify_shard(self.f.partition, 'group1', request['context'], request['raw_seal'],
                self.f.projection, raw_continuation=changed['raw_continuation'], source_migration=self.f.migration_pin)

    def test_failed_continuation_or_changed_archive_fails_before_any_backend(self):
        changed = p.metadata(self.f.descriptor)
        changed['predecessor_adapter_archive'] = dict(self.f.adapter, sha256='0' * 64)
        descriptor = write(self.base / 'archive-descriptor.json', changed)
        with self.assertRaises(ValueError):
            p.prepared_raw_state(self.f.group['new_pin'], self.f.view_pin,
                raw_continuation=descriptor, source_migration=self.f.migration_pin)
        migration = p.metadata(self.f.migration_pin)
        migration['sources']['bench/quality_adaptation_portable.py']['archive']['sha256'] = '0' * 64
        migration_pin = write(self.base / 'changed-source-archive.json', migration)
        with self.assertRaises(ValueError):
            shards.verify_partition(self.f.partition, source_migration=migration_pin)
        receipt = p.metadata(self.f.receipt); receipt['status'] = 'failed'
        changed = p.metadata(self.f.descriptor)
        changed['receipt'] = write(self.base / 'failed-receipt.json', receipt)
        descriptor = write(self.base / 'failed-descriptor.json', changed)
        with self.assertRaises(ValueError):
            p.prepared_raw_state(self.f.group['new_pin'], self.f.view_pin,
                raw_continuation=descriptor, source_migration=self.f.migration_pin)

    def test_real_immutable_facade_uses_authenticated_archived_scanner(self):
        from scripts import kaggle_supervision_v1 as supervision
        adapter = SimpleNamespace(__file__=str(ROOT / 'scripts/kaggle_adaptation.py'), file_sha256=p.sha256)
        request = dict(source_root=str(ROOT), source_migration=self.f.migration_pin)
        facade = child._supervision_facade(request, adapter)
        self.assertIsInstance(facade, supervision.Facade)
        self.assertEqual(p.sha256(Path(facade.scanner.__file__)), supervision.SCANNER_SHA256)
        self.assertNotEqual(Path(facade.scanner.__file__), Path(raw.__file__))
        with self.assertRaisesRegex(ValueError, 'archived original'):
            child._supervision_facade(dict(source_root=str(ROOT)), adapter)
        migration = p.metadata(self.f.migration_pin)
        migration['sources']['bench/quality_adaptation_portable.py']['archive']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            child._supervision_facade(dict(request, source_migration=write(self.base / 'bad-scanner.json', migration)), adapter)

    def test_fresh_interpreter_reconstructs_serialized_child_continuation(self):
        request = self.request(); pin = write(self.base / 'child-request.json', request)
        script = '''
import json, sys
from unittest.mock import patch
from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_portable_prepare as p
from scripts import kaggle_sharded_prepared_v1 as child
request = p.load_request(sys.argv[1], sys.argv[2])
# Only unavailable external asset preflight/resource boundaries are doubled.
with patch.object(raw, 'preflight_inputs', return_value={}), patch.object(raw, 'resource_check', return_value={}):
    proof, state = child._child_raw(request, payloads=False)
assert 'torch' not in sys.modules
print(json.dumps(dict(context=request['context'], continuation=request['raw_continuation'],
    migration=request['source_migration'], ids=[r['id'] for r in state['captures']],
    splits=[r['split'] for r in state['captures']], projection=proof['projection_pin'])))
'''
        result = subprocess.run([sys.executable, '-E', '-s', '-B', '-c', script, pin['path'], pin['sha256']],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout)
        self.assertEqual(actual['context'], request['context'])
        self.assertEqual(actual['continuation'], self.f.descriptor)
        self.assertEqual(actual['migration'], self.f.migration_pin)
        self.assertEqual(actual['projection'], self.f.projection)
        states = p.metadata(p.metadata(self.f.receipt)['capture_index'])['rows']
        self.assertEqual(actual['ids'], [r['id'] for r in states])
        self.assertEqual(actual['splits'], [r['split'] for r in states])

    def test_regular_interpreter_constructor_keeps_legacy_no_links_rule(self):
        regular = Path(sys.executable).resolve()
        with patch.object(p, 'load_source', side_effect=AssertionError('runtime link admission was reached')):
            self.assertEqual(p.prepared_python(regular, {}, {}, self.base), regular)

    @unittest.skipIf(os.name == 'nt', 'mandatory Linux runtime symlink layout integration before cloud use')
    def test_child_request_preserves_only_exact_authenticated_venv_interpreter(self):
        runtime_root = self.base / 'runtime'
        managed = runtime_root / 'python/bin/python3.11'; managed.parent.mkdir(parents=True)
        managed.write_bytes(b'SOFTWARE EXECUTABLE FIXTURE: NEVER LAUNCHED')
        python = runtime_root / 'venv/bin/python'; python.parent.mkdir(parents=True)
        python.symlink_to('../../python/bin/python3.11')
        doc = dict(schema='kaggle_isolated_runtime_v1', runtime_root=str(runtime_root),
            runtime_links=[dict(path='venv/bin/python', target='../../python/bin/python3.11')],
            runtime=dict(executable=str(python), executable_sha256=p.sha256(managed),
                prefix=str(runtime_root / 'venv'), base_prefix=str(runtime_root / 'python')))
        runtime_pin = write(self.base / 'runtime.json', doc)
        request = self.request(python=python, runtime=runtime_pin)
        self.assertEqual(request['python'], str(python))
        self.assertNotEqual(request['python'], str(python.resolve()))
        adapter = p.pin(ROOT / 'scripts/kaggle_adaptation.py')
        with self.assertRaises(ValueError):
            p.prepared_python(python, write(self.base / 'bad-runtime.json',
                dict(doc, runtime=dict(doc['runtime'], executable_sha256='0' * 64))), adapter, self.base)
        extra = runtime_root / 'venv/bin/other'; extra.symlink_to('../../python/bin/python3.11')
        with self.assertRaises(ValueError):
            p.prepared_python(extra, runtime_pin, adapter, self.base)
        with self.assertRaisesRegex(ValueError, 'unapproved runtime links'):
            # The real immutable policy rejects an unlisted link anywhere in
            # the runtime, even if the selected interpreter itself is unchanged.
            p.prepared_python(python, runtime_pin, adapter, self.base)

    @unittest.skipIf(os.name == 'nt', 'mandatory Linux physical hardlink integration before cloud use')
    def test_continuation_aggregate_retained_physical_hardlink_roundtrip(self):
        f = SoftwareJoin(self.base / 'aggregate-fixture', aggregate_only=True)
        partition = p.metadata(f.partition)
        refs = [dict(shard_id='smoke0', context=f.smoke['old_pin'],
            raw_seal=p.pin(f.smoke['root'] / 'raw-seal.json'),
            projection=f.smoke_projection),
            dict(shard_id='group1', context=f.group['new_pin'], raw_seal=f.view_pin, projection=f.projection)]
        for sid, value in f.extra.items():
            projection = shards.seal_shard(f.partition, sid, value['old_pin'],
                p.pin(value['root'] / 'raw-seal.json'), self.base / (sid + '-projection.json'),
                source_migration=f.migration_pin)
            refs.append(dict(shard_id=sid, context=value['old_pin'],
                raw_seal=p.pin(value['root'] / 'raw-seal.json'), projection=projection))
        before = {path: path.read_bytes() for path in f.group['root'].rglob('*.json')}
        aggregate = shards.aggregate_raw(f.partition, refs, self.base / 'aggregate', source_migration=f.migration_pin)
        verified = shards.verify_aggregate_raw(f.partition, aggregate['root'], payloads=True,
            source_migration=f.migration_pin)
        source = next(row for row in p.metadata(p.metadata(f.receipt)['physical_index'])['rows']
            if row['split'] == 'train' and row['bank']['status'] == 'available')
        target = next(row for row in verified['physical'] if row['id'] == source['id'])
        for key in ('bank', 'cnos_bank'):
            old, new = Path(source['bank'][key]['path']), Path(target['bank'][key]['path'])
            self.assertTrue(old.is_relative_to(f.group['root']))
            self.assertFalse(old.is_relative_to(f.attempt / 'preparation-view'))
            self.assertEqual((old.stat().st_dev, old.stat().st_ino), (new.stat().st_dev, new.stat().st_ino))
            self.assertEqual(old.read_bytes(), new.read_bytes())
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(len(verified['captures']), 360)
        self.assertFalse(verified['seal']['training_ready'])
        escaped = write(self.base / 'outside.json', dict(SOFTWARE_FIXTURE=True))
        with self.assertRaisesRegex(ValueError, 'escapes authenticated'):
            shards._link_file(escaped, None, self.base / 'forbidden-link.json', must_be_under=f.group['root'])


if __name__ == '__main__':
    unittest.main()
