"""Mock-only integration checks; passing these is not actual model parity evidence."""
from contextlib import nullcontext
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_adaptation_export_parity as parity
from . import quality_adaptation_data as data
from . import quality_adaptation_runtime as runtime
from . import quality_gotrack_inference as installer
from . import test_quality_gotrack_inference as atomic_doubles


class ArrayDouble:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class HeadDouble(atomic_doubles.AtomicHeadInstallTests.Head):
    calls = []
    output_mode = None

    def __call__(self, tokens, shape):
        if shape != (280, 280) or any(tokens[i] is None for i in data.HOOKS):
            raise AssertionError('actual DPT interface was not used')
        if any(tokens[i] is not None for i in set(range(12)) - set(data.HOOKS)):
            raise AssertionError('unexpected token hook')
        type(self).calls.append(self.device)
        value = float(self.values['projection.weight'].values[0])
        flow = np.full((1, 2, 280, 280), value, dtype=np.float32)
        confidence = np.full((1, 280, 280), .5, dtype=np.float32)
        if type(self).output_mode == 'nonfinite':
            flow[0, 0, 0, 0] = np.nan
        if type(self).output_mode == 'shape':
            confidence = confidence[:, None]
        return ArrayDouble(flow), ArrayDouble(confidence)


class TorchDouble(atomic_doubles.AtomicHeadInstallTests.TorchDouble):
    def __init__(self, export, export_bytes, state, checkpoint_bytes):
        super().__init__(export, export_bytes)
        self.state = state
        self.checkpoint_bytes = checkpoint_bytes
        self.cuda.is_available = lambda: False

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def from_numpy(value):
        return ArrayDouble(value)

    def load(self, stream, *, map_location, weights_only):
        if map_location != 'cpu' or weights_only is not True:
            raise AssertionError('all loads must be CPU weights_only')
        raw = stream.read()
        if raw == self.checkpoint_bytes:
            return self.state
        return super().load(__import__('io').BytesIO(raw), map_location=map_location, weights_only=weights_only)


class ExportParityTests(unittest.TestCase):
    def setUp(self):
        root = parity.ROOT / '.cache/cloud-prepare/test-temp'
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.record = data.read_json(parity.MANIFEST)
        self.export_bytes = b'mock-only-export-no-model'
        self.checkpoint_bytes = b'mock-only-reference-no-model'
        self.export_path = self.root / 'head.pt'
        self.export_path.write_bytes(self.export_bytes)
        self.checkpoint_path = self.root / 'best.pt'
        self.checkpoint_path.write_bytes(self.checkpoint_bytes)
        self.record.update(export_sha256=hashlib.sha256(self.export_bytes).hexdigest(),
            export_bytes=len(self.export_bytes),
            reference_checkpoint_sha256=hashlib.sha256(self.checkpoint_bytes).hexdigest(),
            reference_checkpoint_bytes=len(self.checkpoint_bytes))
        self.versions = {'mocked': 'no neural execution'}
        self.dataset_path = self.make_dataset()
        self.record['dataset_sha256'] = data.canonical_hash(data.read_json(self.dataset_path))
        self.write_manifest()
        self.heads = []
        HeadDouble.calls = []
        HeadDouble.output_mode = None

    def write_manifest(self, raw=None):
        self.manifest = self.root / 'artifact.json'
        self.manifest.write_bytes(raw if raw is not None else json.dumps(self.record).encode())
        self.manifest_hash = data.sha256(self.manifest)

    def make_dataset(self):
        recipe = {'schema': 'frozen_gotrack_decoder_recipe_v1', 'annotation_inputs': [],
            'source_lock_sha256': data.sha256(runtime.LOCK), 'hooks': list(data.HOOKS),
            'producer_sha256': data.sha256(parity.ROOT / 'bench/quality_adaptation_prepare.py')}
        recipe_hash = data.canonical_hash(recipe)
        (self.root / 'recipe.json').write_text(json.dumps(recipe), encoding='utf-8')
        rows = []
        for key, split, status, object_id in [('train', 'train', 'available', '1'),
                ('val-z', 'val', 'available', '2'), ('val-a', 'val', 'available', '2'),
                ('val-unavailable', 'val', 'unavailable', '2'), ('test-trap', 'test', 'unavailable', '3')]:
            row = {'id': key, 'split': split, 'status': status, 'object_id': object_id,
                   'sequence_id': split, 'capture_group': split,
                   'query_rgb_sha256': hashlib.sha256(key.encode()).hexdigest()}
            if status == 'available':
                row.update(features={'path': key + '-features.npz', 'bytes': 10,
                                     'sha256': hashlib.sha256((key + '-feature').encode()).hexdigest()},
                           labels={'path': key + '-LABEL-TRAP.npz', 'bytes': 10, 'sha256': 'f' * 64})
            else:
                row['failure_reason'] = 'preserved_unavailable'
            rows.append(row)
        (self.root / 'index.json').write_text(json.dumps({'recipe_sha256': recipe_hash, 'rows': rows}), encoding='utf-8')
        manifest = {'schema': data.SCHEMA, 'conventions': data.CONVENTION,
                    'provenance': {'feature_recipe_sha256': recipe_hash, 'source_lock_sha256': data.sha256(runtime.LOCK),
                                   'annotation_inputs': [], 'label_origin': 'synthetic_renderer_training_only'}, 'rows': rows}
        target = self.root / 'dataset.json'
        target.write_text(json.dumps(manifest), encoding='utf-8')
        return target

    def exported(self):
        return {'schema': self.record['export_schema'], 'head': {'projection.weight': atomic_doubles.AtomicHeadInstallTests.Tensor(np.array([1., 2.], dtype=np.float32))},
            'source_lock_sha256': self.record['training_lock_sha256'],
            'base_checkpoint_sha256': self.record['base_checkpoint_sha256'],
            'dataset_sha256': self.record['dataset_sha256'], 'config_sha256': self.record['training_config_sha256'],
            'validation': {'mocked': True}, 'quality_claim': False, 'deployment': self.record['export_deployment']}

    def state(self):
        return {'schema': 'correspondence_head_checkpoint_v1', 'dataset_sha256': self.record['dataset_sha256'],
            'config_sha256': self.record['training_config_sha256'], 'source_lock_sha256': self.record['training_lock_sha256'],
            'versions': self.versions, 'epoch': 1, 'batch_cursor': 0, 'steps': 1, 'bad_epochs': 0,
            'elapsed_seconds': 1., 'head': copy.deepcopy(self.exported()['head'])}

    def context(self):
        return parity.context_for_export(self.record, self.manifest_hash, self.export_path)

    def load_head(self, *, initialize, device):
        self.assertFalse(initialize)
        self.assertEqual(device, 'cpu')
        head = HeadDouble()
        self.heads.append(head)
        return head

    def features(self, ds, row, device, torch):
        self.assertEqual(row['split'], 'val')
        self.assertNotIn('TRAP', row['features']['path'])
        return [ArrayDouble(np.zeros((1, 400, 768), dtype=np.float32)) if i in data.HOOKS else None for i in range(12)]

    def invoke(self, *, export=None, state=None, max_rows=1):
        torch = TorchDouble(self.exported() if export is None else export, self.export_bytes,
                            self.state() if state is None else state, self.checkpoint_bytes)
        with patch.object(runtime, 'cloud_guard') as guard, patch.object(runtime, 'runtime_versions', return_value=self.versions), \
                patch.object(runtime, 'verify_sources', return_value={'source_lock_sha256': self.record['training_lock_sha256'],
                    'upstream_revision': self.record['upstream_revision']}), \
                patch.object(runtime, 'load_head', side_effect=self.load_head), \
                patch.object(parity, 'feature_tokens', side_effect=self.features), \
                patch.object(data.Dataset, 'arrays', side_effect=AssertionError('LABEL/TEST payload opened')), \
                patch.dict(sys.modules, {'torch': torch}):
            receipt = parity.run(artifact_manifest=self.manifest, artifact_manifest_sha256=self.manifest_hash,
                export=self.export_path, reference_checkpoint=self.checkpoint_path, dataset=self.dataset_path,
                output=self.root / 'receipt', deadline_utc=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat().replace('+00:00', 'Z'),
                max_seconds=20, max_rows=max_rows, cloud_execution=True)
            guard.assert_called_once_with(True)
        return receipt, torch

    def test_actual_installer_path_with_mocked_heads_and_no_label_test_reads(self):
        receipt, torch = self.invoke()
        self.assertEqual(receipt['status'], 'passed', receipt)
        self.assertTrue(receipt['cached_token_parity_passed'])
        self.assertEqual(receipt['selected_row_ids'], ['val-a'])
        self.assertEqual(receipt['validation_inventory'], {'total': 3, 'available': 2, 'unavailable': 1,
            'selected': 1, 'selection': 'first_available_VAL_ids_sorted_lexically'})
        self.assertEqual(HeadDouble.calls, ['cpu', 'cpu'])
        self.assertEqual(receipt['comparisons'][0]['flow_max_absolute_difference_px'], 0.)
        self.assertTrue(receipt['installation']['head_installation']['installed'])
        for key in ('full_network_forward_executed', 'pose_replay', 'quality_claim'):
            self.assertIs(receipt[key], False)
        self.assertEqual(receipt['label_payload_reads'], 0)
        self.assertEqual(receipt['test_payload_reads'], 0)
        self.assertEqual(torch.load_calls, 1)
        retained = data.read_json(self.root / 'receipt/receipt.json')
        self.assertEqual(retained, receipt)
        self.assertLess((self.root / 'receipt/receipt.json').stat().st_size, parity.MAX_RECEIPT)
        with self.assertRaises(FileExistsError):
            parity.run(output=self.root / 'receipt', artifact_manifest=None, artifact_manifest_sha256=None,
                export=None, reference_checkpoint=None, dataset=None, deadline_utc=None, max_seconds=1)

    def test_manifest_bad_digest_unknown_fields_duplicate_keys_and_bad_schema(self):
        with self.assertRaisesRegex(ValueError, 'digest'):
            parity.read_manifest(self.manifest, '0' * 64)
        for change in ({'schema': 'wrong'}, {'unreviewed_code_path': 'evil.py'}, {'quality_claim': True}, {'export_bytes': True}):
            original = self.record.copy()
            self.record.update(change)
            self.write_manifest()
            with self.assertRaises(ValueError):
                parity.read_manifest(self.manifest, self.manifest_hash)
            self.record = original
        self.write_manifest(b'{"schema":1,"schema":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            parity.read_manifest(self.manifest, self.manifest_hash)

    def test_loaded_installer_source_is_authenticated(self):
        self.record['installer_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'installer source'):
            self.context()

    def test_checkpoint_metadata_export_header_and_state_fail_closed(self):
        for kind in ('checkpoint', 'header', 'state'):
            with self.subTest(kind=kind):
                export, state = self.exported(), self.state()
                if kind == 'checkpoint':
                    state['config_sha256'] = '0' * 64
                elif kind == 'header':
                    export['dataset_sha256'] = '0' * 64
                else:
                    export['head'] = {}
                receipt, _ = self.invoke(export=export, state=state)
                self.assertEqual(receipt['status'], 'failed')
                self.assertFalse(receipt['cached_token_parity_passed'])
                # Each subcase retains its failure in a distinct new directory.
                (self.root / 'receipt').rename(self.root / ('failure-' + kind))

    def test_corrupt_export_or_checkpoint_never_constructs_a_head(self):
        for name in ('export', 'checkpoint'):
            target = self.export_path if name == 'export' else self.checkpoint_path
            original = target.read_bytes()
            target.write_bytes(b'corrupt')
            receipt, _ = self.invoke()
            self.assertEqual(receipt['status'], 'failed')
            self.assertRegex(receipt['error']['message'], 'digest or size mismatch|exceeds')
            self.assertEqual(self.heads, [])
            target.write_bytes(original)
            (self.root / 'receipt').rename(self.root / ('corrupt-' + name))

    def test_wrong_dataset_identity_opens_no_arrays_or_heads(self):
        self.record['dataset_sha256'] = '0' * 64
        self.write_manifest()
        receipt, _ = self.invoke()
        self.assertEqual(receipt['status'], 'failed')
        self.assertRegex(receipt['error']['message'], 'dataset identity')
        self.assertEqual(self.heads, [])

    def test_atomic_install_device_failure_retains_live_head_and_rng(self):
        network = SimpleNamespace(pose_head=HeadDouble(fail_device=True))
        original = network.pose_head
        torch = TorchDouble(self.exported(), self.export_bytes, self.state(), self.checkpoint_bytes)
        with self.assertRaisesRegex(RuntimeError, 'device preparation'):
            installer.install_adapted_head(network, self.context(), device='cuda', torch_module=torch)
        self.assertIs(network.pose_head, original)
        self.assertEqual(original.load_calls, 0)
        self.assertEqual(torch.cpu_state.value, 'cpu-before')
        self.assertEqual(torch.cuda_state[0].value, 'cuda-before')

    def test_shape_nonfinite_and_parity_failure_are_retained(self):
        for mode in ('shape', 'nonfinite', 'parity'):
            HeadDouble.output_mode = mode
            export = self.exported()
            if mode == 'parity':
                export['head']['projection.weight'] = atomic_doubles.AtomicHeadInstallTests.Tensor(np.array([2., 3.], dtype=np.float32))
            # Keep the reference checkpoint independent from deliberately changed export.
            state = self.state()
            receipt, _ = self.invoke(export=export, state=state)
            self.assertEqual(receipt['status'], 'failed', receipt)
            self.assertFalse(receipt['cached_token_parity_passed'])
            self.assertRegex(receipt['error']['message'], 'output shape/dtype/nonfinite|parity failed')
            (self.root / 'receipt').rename(self.root / ('failure-' + mode))

    def test_feature_loader_only_opens_verified_available_val_feature_keys(self):
        ds = data.load_dataset(self.dataset_path, inspect_arrays=False)
        row = ds.rows('val')[0]
        values = {f'tokens_{i}': np.zeros((400, 768), dtype=np.float32) for i in data.HOOKS}
        values.update(template_mask=np.ones((280, 280), dtype=bool),
            template_xyz_m=np.zeros((280, 280, 3), dtype=np.float32),
            query_K=np.array([[280, 0, 140], [0, 280, 140], [0, 0, 1]], dtype=np.float64),
            baseline_flow=np.zeros((2, 280, 280), dtype=np.float32),
            baseline_confidence=np.full((280, 280), .5, dtype=np.float32))
        with patch.object(data, 'descriptor', return_value=self.root / 'only-features.npz') as descriptor, \
                patch.object(data, 'load_npz', return_value=values) as load, \
                patch.object(data.Dataset, 'arrays', side_effect=AssertionError('labels opened')):
            tokens = parity.feature_tokens(ds, row, 'cpu', TorchDouble(None, b'', None, b''))
            descriptor.assert_called_once_with(ds.root, row['features'], verify=True)
            load.assert_called_once_with(self.root / 'only-features.npz', data.FEATURE_KEYS)
            self.assertEqual(tokens[2].value.shape, (1, 400, 768))
            with self.assertRaisesRegex(ValueError, 'only available VAL'):
                parity.feature_tokens(ds, ds.rows('test', available=False)[0], 'cpu', None)
            self.assertEqual(load.call_count, 1)

    def test_expired_deadline_and_budget_bounds_start_no_work(self):
        with self.assertRaises(TimeoutError):
            parity.Budget('2000-01-01T00:00:00Z', 10)
        for seconds in (True, 0, 901, float('nan')):
            with self.assertRaises(ValueError):
                parity.Budget('2100-01-01T00:00:00Z', seconds)
        with patch.object(parity.time, 'monotonic', side_effect=[0., 0., 20.]):
            budget = parity.Budget('2100-01-01T00:00:00Z', 10)
            with self.assertRaises(TimeoutError):
                budget.check()
        for count in (0, 9, True):
            with self.assertRaises(ValueError):
                parity.select_validation(None, count)


if __name__ == '__main__':
    unittest.main()
