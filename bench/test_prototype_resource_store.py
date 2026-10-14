"""Synthetic byte/isolation/parity controls; no dataset or model quality claim."""
import binascii
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

from .prototype_resource_store import (ResourceStore, provider, sha_file,
                                      verified_stream, safe_relative, strict_json)

REVISION = '985ebaef7b1bd4bffb2039f18dd81faa647fd060'


def payload(raw, method=8):
    if method == 0:
        return raw
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(raw) + compressor.flush()


def row(raw, compressed, name, role):
    return dict(datasetRevision=REVISION, archive='fixture.zip', sourceMember=name,
        method=8, sourceMethod=8, sourceFlags=0, sourceCompressedBytes=len(compressed),
        expandedBytes=len(raw), compressedBytes=len(compressed),
        crc32=binascii.crc32(raw) & 0xffffffff,
        compressedSHA256=hashlib.sha256(compressed).hexdigest(),
        uncompressedSHA256=hashlib.sha256(raw).hexdigest(), allowedRoles=[role],
        location={'kind': 'payload', 'path': 'payloads/' + hashlib.sha256(name.encode()).hexdigest() + '.bin'})


def make_store(root, assets):
    root.mkdir(exist_ok=True)
    entries, specs = {}, {}
    for name, (raw, role) in assets.items():
        compressed = payload(raw)
        entries[name] = row(raw, compressed, name, role)
        path = root / entries[name]['location']['path']
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(compressed)
        specs[name] = {'path': name, 'sha256': entries[name]['uncompressedSHA256']}
    index = dict(schemaVersion=1, corpus='HB', datasetRevision=REVISION,
                 selectionDigest='a' * 64, storageKind='original_zip_members', resources=entries)
    (root / 'resources.json').write_text(json.dumps(index), encoding='utf-8')
    storage = dict(indexPath='resources.json', indexSHA256=sha_file(root / 'resources.json'),
                   datasetRevision=REVISION, selectionDigest='a' * 64, corpus='HB')
    return storage, specs, entries


class ResourceTests(unittest.TestCase):
    def test_exact_bytes_seek_text_and_multichunk_expansion(self):
        raw = (b'ASCII PLY control\n' * 180000) + bytes(range(256))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            storage, specs, _ = make_store(root, {'models/control.ply': (raw, 'model')})
            reader = ResourceStore(root, storage, 'inference', {('models/control.ply', 'model')})
            with reader.open_binary(specs['models/control.ply'], 'model') as stream:
                self.assertEqual(stream.read(), raw)
                stream.seek(20)
                self.assertEqual(stream.read(100), raw[20:120])
            text = b'ply\nformat ascii 1.0\n'
            storage, specs, _ = make_store(root, {'models/control.ply': (text, 'model')})
            reader = ResourceStore(root, storage, 'evaluation')
            with reader.open_text(specs['models/control.ply'], 'model') as source:
                self.assertEqual(source.read(), text.decode('ascii'))

    def test_corruption_truncation_trailing_crc_and_metadata_fail_closed(self):
        raw = b'Control reference bytes' * 200000
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'payload'
            compressed = payload(raw)
            original = row(raw, compressed, 'rgb/control.png', 'rgb')
            for damaged in (compressed[:-1], compressed + b'extra', bytes([compressed[0] ^ 1]) + compressed[1:]):
                path.write_bytes(damaged)
                changed = {**original, 'compressedBytes': len(damaged),
                           'compressedSHA256': hashlib.sha256(damaged).hexdigest()}
                with self.assertRaises((ValueError, zlib.error)):
                    verified_stream(path, changed)
            path.write_bytes(compressed)
            for key, value in (('crc32', original['crc32'] ^ 1), ('expandedBytes', len(raw) - 1),
                               ('uncompressedSHA256', '0' * 64), ('compressedSHA256', '0' * 64), ('method', 12)):
                with self.assertRaises((ValueError, zlib.error)):
                    verified_stream(path, {**original, key: value})

    def test_phase_whitelist_and_index_mutation_before_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            storage, specs, _ = make_store(root, {'rgb/frame.png': (b'RGB', 'rgb'),
                'depth/scored.png': (b'label', 'evaluation_depth'),
                'mask/scored.png': (b'mask', 'evaluation_mask')})
            log = []
            reader = ResourceStore(root, storage, 'inference', {('rgb/frame.png', 'rgb')}, log)
            with patch('bench.prototype_resource_store.verified_stream', wraps=verified_stream) as spy:
                for name, role in (('depth/scored.png', 'evaluation_depth'), ('mask/scored.png', 'rgb')):
                    with self.assertRaises(ValueError):
                        reader.open_binary(specs[name], role)
                spy.assert_not_called()
                with reader.open_binary(specs['rgb/frame.png'], 'rgb') as stream:
                    self.assertEqual(stream.read(), b'RGB')
            self.assertEqual(log, [dict(phase='inference', resource='rgb/frame.png', role='rgb', sha256=specs['rgb/frame.png']['sha256'])])
            gallery = ResourceStore(root, storage, 'gallery')
            with patch('bench.prototype_resource_store.verified_stream', wraps=verified_stream) as spy:
                with self.assertRaises(ValueError):
                    gallery.open_binary(specs['mask/scored.png'], 'evaluation_mask')
                spy.assert_not_called()  # Gallery visualization never consumes evaluation masks.
                with gallery.open_binary(specs['depth/scored.png'], 'evaluation_depth') as stream:
                    self.assertEqual(stream.read(), b'label')
            (root / 'resources.json').write_text('{}')
            with self.assertRaises(ValueError):
                reader.open_binary(specs['rgb/frame.png'], 'rgb')

    def test_unsafe_paths_duplicate_keys_and_identity(self):
        for value in ('../file', '/file', 'a\\b', 'C:/file', 'a/./b', 'a//b'):
            with self.assertRaises(ValueError):
                safe_relative(value)
        with self.assertRaises(ValueError):
            strict_json('{"resources": {}, "resources": {}}')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            storage, _, entries = make_store(root, {'rgb/a.png': (b'RGB', 'rgb')})
            entries['rgb/b.png'] = copy.deepcopy(entries['rgb/a.png'])
            index = json.loads((root / 'resources.json').read_text())
            index['resources'] = entries
            (root / 'resources.json').write_text(json.dumps(index))
            storage['indexSHA256'] = sha_file(root / 'resources.json')
            with self.assertRaises(ValueError):
                ResourceStore(root, storage, 'evaluation')

    def test_original_pixels_depth_mask_and_ply_geometry(self):
        import numpy as np
        from PIL import Image
        from .ply_model import read_ply
        arrays = {'rgb/x.png': np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3),
                  'depth/x.png': np.arange(48, dtype=np.uint16).reshape(6, 8),
                  'mask/x.png': np.tile([0, 255], (6, 4)).astype(np.uint8)}
        assets = {}
        for name, array in arrays.items():
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format='PNG')
            assets[name] = (stream.getvalue(), {'rgb': 'rgb', 'depth': 'evaluation_depth', 'mask': 'evaluation_mask'}[name.split('/')[0]])
        ply = b'ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n100 0 0\n0 100 0\n3 0 1 2\n'
        assets['models/x.ply'] = (ply, 'model')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            storage, specs, _ = make_store(root, assets)
            reader = ResourceStore(root, storage, 'evaluation')
            for name, expected in arrays.items():
                with reader.open_binary(specs[name], assets[name][1]) as source, Image.open(source) as image:
                    np.testing.assert_array_equal(np.asarray(image), expected)
            direct = root / 'original.ply'
            direct.write_bytes(ply)
            extracted = read_ply(direct, 'control')
            compact = read_ply(reader.open_binary(specs['models/x.ply'], 'model'), 'control')
            for left, right in zip(extracted[1:], compact[1:]):
                np.testing.assert_array_equal(left, right)
            # Analytic fixture: input millimetres multiplied by 0.001 yields these metre vertices.
            np.testing.assert_array_equal(compact[1], [[0, 0, 0], [.1, 0, 0], [0, .1, 0]])

    def test_hb_gallery_and_legacy_ycb_labels(self):
        from .prototype_gallery import _evidence_contract, GalleryError
        from .prototype_benchmark import score_trace
        legacy = dict(evidenceKind='dataset_reference', benchmarkMode='full', realUnseenObjects=10)
        self.assertEqual(_evidence_contract(legacy, legacy, 10)['objectIdLabel'], 'YCBV object ID')
        hb = dict(legacy, benchmarkMode='followup', datasetName='HB', objectIdLabel='HB ID',
                  provenance='HB initialized tracking transfer; pretrained exposure unknown; no human truth')
        self.assertEqual(_evidence_contract(hb, hb, 10)['objectIdLabel'], 'HB ID')
        self.assertEqual(_evidence_contract(hb, hb, 10)['benchmarkMode'], 'followup')
        with self.assertRaises(GalleryError):
            _evidence_contract(dict(hb, datasetName='YCB'), hb, 10)
        records = [dict(frameId=1, evaluationInfo={'visib_fract': 1})]
        frames = [dict(frameId=1, state='lost', failureReason='control', cvCameraFromSourceObject=None)]
        self.assertIn('BOP/HB', score_trace(records, frames, {}, 'HB')['provenance'])
        self.assertIn('BOP/YCB-Video', score_trace(records, frames, {})['provenance'])

    def test_invalid_inference_seal_rejects_before_opening_evaluator_labels(self):
        from .prototype_benchmark import run_evaluation
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, output = root / 'dataset', root / 'output'
            dataset.mkdir()
            output.mkdir()
            (dataset / 'evaluator.json').write_text('{"privateScoredLabels": true}')
            (output / 'freeze.json').write_text('{}')
            (output / 'inference-seal.json').write_text(json.dumps(dict(
                evaluatorManifestSHA256='0' * 64, freezeSHA256='0' * 64)))
            original_open = Path.open
            accesses = []
            def spy(path, *args, **kwargs):
                accesses.append(path.name)
                return original_open(path, *args, **kwargs)
            with patch.object(Path, 'open', spy):
                with self.assertRaisesRegex(ValueError, 'freeze invalidated'):
                    run_evaluation(dataset, output)
            self.assertNotIn('evaluator.json', accesses)

    def test_reuse_is_hash_bound_and_outside_allowlist_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / 'metadata'
            metadata.mkdir()
            local = metadata / 'models/models_info.json'
            local.parent.mkdir()
            raw = b'{"1": {"diameter": 100}}'
            local.write_bytes(raw)
            storage, specs, entries = make_store(root, {'models/models_info.json': (raw, 'model_info')})
            entry = entries['models/models_info.json']
            entry.update(method=0, compressedBytes=len(raw), compressedSHA256=entry['uncompressedSHA256'],
                         location={'kind': 'reuse', 'path': str(local)})
            index = json.loads((root / 'resources.json').read_text())
            index['resources'] = entries
            (root / 'resources.json').write_text(json.dumps(index))
            storage['indexSHA256'] = sha_file(root / 'resources.json')
            with patch('bench.prototype_resource_store.REUSE_ROOT', metadata):
                reader = ResourceStore(root, storage, 'inference', {('models/models_info.json', 'model_info')})
                with reader.open_binary(specs['models/models_info.json'], 'model_info') as stream:
                    self.assertEqual(stream.read(), raw)
                local.write_bytes(raw.replace(b'100', b'101'))
                with self.assertRaises(ValueError):
                    reader.open_binary(specs['models/models_info.json'], 'model_info')
            with self.assertRaises(ValueError):
                ResourceStore(root, storage, 'inference', {('models/models_info.json', 'model_info')})


class BenchmarkParityTests(unittest.TestCase):
    def test_original_snapshot_extracted_and_compact_inference_parity_and_label_spy(self):
        """A deterministic RGB-consuming fake tracker is an interface control.

        Expected identity rotation/translation and forced losses are analytic
        control outputs, not fabricated real tracking or model-quality truth.
        """
        import numpy as np
        from PIL import Image
        from . import prototype_benchmark as current, prototype_tracking
        from .prototype_output_budget import CompactOutputBudget
        baseline_path = current.ROOT / '.cache/prototype-sprint/results-v1/source-snapshot/bench/prototype_benchmark.py'
        self.assertTrue(baseline_path.is_file(), 'Retained authoritative baseline snapshot required')
        spec = importlib.util.spec_from_file_location('bench._compact_parity_baseline', baseline_path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)

        class Result:
            def __init__(self, record):
                self.record = record
            def manifest(self):
                return dict(self.record)

        class ControlledTracker:
            def __init__(self, *args, **kwargs):
                pass
            def reset_temporal(self):
                pass
            def update(self, packet):
                value = int(packet.rgb[0, 0, 0])
                matrix = np.eye(4)
                matrix[0, 3] = value * .001
                return Result(dict(frameId=packet.frame_id, state='tracking' if value else 'lost',
                    failureReason=None if value else 'analytic_blank_control',
                    cameraFromObject=matrix.tolist() if value else None))

        ply = b'ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n100 0 0\n0 100 0\n3 0 1 2\n'
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            extracted, compact = base / 'extracted', base / 'compact'
            extracted.mkdir()
            stream = io.BytesIO()
            Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(stream, format='PNG')
            assets = {'models/x.ply': (ply, 'model'), 'depth/onboard.png': (stream.getvalue(), 'onboarding_depth'),
                      'depth/scored.png': (stream.getvalue(), 'evaluation_depth'), 'mask/scored.png': (stream.getvalue(), 'evaluation_mask')}
            for frame_id in range(120):
                stream = io.BytesIO()
                Image.fromarray(np.full((4, 4, 3), frame_id + 1, dtype=np.uint8)).save(stream, format='PNG')
                assets[f'rgb/{frame_id:06d}.png'] = (stream.getvalue(), 'rgb')
            storage, specs, _ = make_store(compact, assets)
            for name, (raw, _) in assets.items():
                path = extracted / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(raw)
            calibration = dict(cam_K=[100, 0, 2, 0, 100, 2, 0, 0, 1], depth_scale=1)
            records = [dict(frameId=i, logicalIndex=i, rgb=specs[f'rgb/{i:06d}.png'], calibration=calibration)
                       for i in range(120)]
            onboarding = [dict(r, depth=specs['depth/onboard.png'], evaluationPose={}) for r in records[:10]]
            manifest = dict(benchmarkId='analytic-control', selectionDigest='a' * 64,
                evaluatorManifestSHA256='e' * 64, dataset='BOP HB val_primesense', evidenceKind='dataset_reference',
                benchmarkMode='followup', objects=[dict(objectId=1, name='control', model=specs['models/x.ply'],
                    onboarding=onboarding, frames=records[10:])])
            (extracted / 'inference.json').write_text(json.dumps(manifest))
            (compact / 'inference.json').write_text(json.dumps(dict(manifest, resourceStorage=storage)))
            (compact / 'evaluator.json').write_text('{}')
            (compact / 'acquisition-receipt.json').write_text(json.dumps(dict(status='complete',
                datasetRevision=REVISION, selectionDigest='a' * 64, indexSHA256=storage['indexSHA256'],
                inferenceManifestSHA256=sha_file(compact / 'inference.json'), evaluatorManifestSHA256='e' * 64)))
            references = (np.zeros((12, 3)), np.zeros((12, 32), dtype=np.uint8), np.zeros((4, 4)),
                          np.zeros((12, 2), dtype=np.float32), np.zeros((12, 3)))
            access_log = []
            original_open = Path.open
            def inference_read_spy(path, *args, **kwargs):
                if path.name == 'evaluator.json':
                    raise AssertionError('Scored evaluator manifest opened during inference')
                return original_open(path, *args, **kwargs)
            original_provider = current.provider
            def logging_provider(root, value, phase):
                return original_provider(root, value, phase, access_log)
            outputs = []
            with patch.object(prototype_tracking, 'PrototypeTrackerV1', ControlledTracker), \
                 patch.object(Path, 'open', inference_read_spy):
                for implementation, dataset in ((baseline, extracted), (current, extracted), (current, compact)):
                    output = base / f'output-{len(outputs)}'
                    with patch.object(implementation, 'build_bank', return_value=references), \
                         patch.object(implementation, 'freeze_source', return_value={}), \
                         patch.object(implementation, 'runtime_identity', return_value={}), \
                         patch.object(current, 'provider', side_effect=logging_provider), \
                         patch.object(current, 'CompactOutputBudget', side_effect=lambda cache, result:
                             CompactOutputBudget(cache, result, sprint_root=base, free_probe=lambda: 4 * 1024**3)):
                        implementation.run_inference(dataset, output)
                    trials = {}
                    for name in ('natural', 'interrupted'):
                        frames = json.loads((output / 'object-000001' / f'{name}.json').read_text())['frames']
                        trials[name] = [{k: v for k, v in f.items() if k != 'processingSeconds'} for f in frames]
                    outputs.append(trials)
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[0], outputs[2])
            self.assertEqual(len(outputs[2]['natural']), 110)
            # Analytic controlled tracker: frame 10 has RGB value 11 and x translation = value * 0.001 metres.
            self.assertEqual(outputs[2]['natural'][0]['cvCameraFromSourceObject'][0][3], .011)
            # Analytic controlled tracker: the five forced zero RGB inputs yield five pose-free losses.
            self.assertEqual([f['state'] for f in outputs[2]['interrupted'][50:55]], ['lost'] * 5)
            self.assertTrue(access_log)
            self.assertTrue(all(row['role'] in ('model', 'rgb', 'onboarding_depth') for row in access_log))
            self.assertFalse(any('scored' in row['resource'] for row in access_log))


class CompactOutputTests(unittest.TestCase):
    def setup_budget(self, root, free=None):
        from .prototype_output_budget import CompactOutputBudget
        cache = root / 'cache'
        cache.mkdir(exist_ok=True)
        (cache / 'pinned-input').write_bytes(b'input')
        return CompactOutputBudget(cache, root / 'results', sprint_root=root,
            free_probe=free or (lambda: 4 * 1024**3))

    def test_actual_json_writer_drop_atomic_peak_and_retained_partial_retry(self):
        from . import prototype_output_budget as ob
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            free = [4 * 1024**3]
            budget = self.setup_budget(root, lambda: free[0])
            destination = budget.result / 'progress.json'
            budget.write_json(destination, {'original': True})
            original = destination.read_bytes()
            # Analytic ledger: result dir + old JSON + new JSON partial need three allocation units.
            with patch.object(ob, 'RESULT_CAP', 2 * ob.UNIT):
                with self.assertRaises(ob.CompactBudgetExceeded):
                    budget.write_json(destination, {'replacement': True})
            self.assertEqual(destination.read_bytes(), original)
            free[0] = ob.FREE_FLOOR - 1
            with self.assertRaises(ob.CompactBudgetExceeded):
                budget.write_json(budget.result / 'next.json', {})
            self.assertFalse((budget.result / 'next.json').exists())
            free[0] = 4 * 1024**3
            writer, target = budget.open_partial(budget.result / 'retained.json')
            writer.write(b'partial retained')
            writer.close()
            self.assertTrue((budget.result / 'retained.json.part').exists())
            with self.assertRaises(ob.CompactBudgetExceeded):
                budget.write_json(target, {})
            with patch.object(ob, 'SCRATCH_CAP', ob.UNIT - 1):
                with self.assertRaises(ob.CompactBudgetExceeded):
                    budget.check()

    def test_first_partial_write_rechecks_physical_allocation_after_open(self):
        from . import prototype_output_budget as ob
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            free = [4 * 1024**3]
            budget = self.setup_budget(root, lambda: free[0])
            writer, destination = budget.open_partial(budget.result / 'first-write.json')
            free[0] = ob.FREE_FLOOR + ob.UNIT - 1
            try:
                with self.assertRaises(ob.CompactBudgetExceeded):
                    writer.write(b'first physical allocation')
            finally:
                writer.close()
            self.assertFalse(destination.exists())
            self.assertEqual(destination.with_name(destination.name + '.part').stat().st_size, 0)

    def test_real_seekable_npz_growth_caps_and_copy_and_gallery_shared_usage(self):
        import numpy as np
        from . import prototype_output_budget as ob
        from .prototype_gallery import _write_limited
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = self.setup_budget(root)
            budget.mkdir(budget.result)
            data = np.random.default_rng(17).integers(0, 256, size=20000, dtype=np.uint8)
            for cap in ('RESULT_CAP', 'TOTAL_CAP'):
                target = budget.result / f'{cap}.npz'
                current = ob.ledger(budget.result if cap == 'RESULT_CAP' else root)
                with patch.object(ob, cap, current + 2 * ob.UNIT):
                    with self.assertRaises(ob.CompactBudgetExceeded):
                        budget.npz(target, data=data)
                self.assertFalse(target.exists())
                self.assertTrue(target.with_name(target.name + '.part').exists())
            copy_source = root / 'source.bin'
            copy_source.write_bytes(b'exact source bytes')
            budget.copy(copy_source, budget.result / 'source-snapshot' / 'source.bin')
            self.assertEqual((budget.result / 'source-snapshot/source.bin').read_bytes(), copy_source.read_bytes())
            gallery = budget.result / 'demo'
            budget.mkdir(gallery)
            _write_limited(gallery, 'assets/control.jpg', b'control', guard=budget)
            current = ob.ledger(budget.result)
            with patch.object(ob, 'RESULT_CAP', current):
                with self.assertRaises(ob.CompactBudgetExceeded):
                    _write_limited(gallery, 'gallery.json', b'{}', guard=budget)
            with patch.object(ob, 'TOTAL_CAP', ob.ledger(root)):
                with self.assertRaises(ob.CompactBudgetExceeded):
                    _write_limited(gallery, 'index.html', b'completion marker', guard=budget)
            self.assertFalse((gallery / 'gallery.json').exists())
            self.assertFalse((gallery / 'index.html').exists())
            self.assertTrue((gallery / 'assets/control.jpg').is_file())
            with self.assertRaises(ob.CompactBudgetExceeded):
                budget.write_bytes(root / 'unaccounted-gallery.json', b'{}')

    def test_compact_infer_below_floor_has_no_output_and_npz_budget_abort_has_no_seal(self):
        import numpy as np
        from types import SimpleNamespace
        from . import prototype_benchmark as benchmark, prototype_output_budget as ob
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / 'cache'
            cache.mkdir()
            records = [dict(frameId=i, rgb={'path': f'rgb/{i}.png', 'sha256': 'a' * 64}, calibration={}) for i in range(10, 120)]
            manifest = dict(resourceStorage={}, dataset='BOP HB val_primesense', benchmarkId='control',
                selectionDigest='a' * 64, evaluatorManifestSHA256='e' * 64,
                objects=[dict(objectId=1, name='analytic I/O control', model={}, onboarding=[{}] * 10, frames=records)])
            (cache / 'inference.json').write_text(json.dumps(manifest))
            reader = object.__new__(ResourceStore)
            reader.phase, reader.access_log = 'inference', []
            reader.assert_sealed = lambda: None
            def guard_factory(cache_path, output):
                return ob.CompactOutputBudget(cache_path, output, sprint_root=root, free_probe=lambda: ob.FREE_FLOOR - 1)
            output = root / 'below-floor'
            with patch.object(benchmark, 'provider', return_value=reader), \
                 patch.object(benchmark, 'CompactOutputBudget', side_effect=guard_factory):
                with self.assertRaises(ob.CompactBudgetExceeded):
                    benchmark.run_inference(cache, output)
            self.assertFalse(output.exists())
            output = root / 'npz-blocked'
            refs = (np.random.default_rng(17).random((2000, 3)), np.zeros((2000, 32), dtype=np.uint8),
                    None, np.zeros((2000, 2)), np.zeros((2000, 3)))
            mesh = SimpleNamespace(positions=np.zeros((3, 3)), normals=np.zeros((3, 3)),
                                   triangles=np.array([[0, 1, 2]]), uv=np.zeros((3, 2)))
            scan = SimpleNamespace(original=mesh)
            def guard_factory(cache_path, result):
                return ob.CompactOutputBudget(cache_path, result, sprint_root=root, free_probe=lambda: 4 * 1024**3)
            with patch.object(benchmark, 'provider', return_value=reader), \
                 patch.object(benchmark, 'CompactOutputBudget', side_effect=guard_factory), \
                 patch.object(benchmark, 'freeze_source', return_value={}), \
                 patch.object(benchmark, 'runtime_identity', return_value={}), \
                 patch.object(benchmark, 'resource', return_value=None), \
                 patch.object(benchmark, 'read_ply', return_value=(scan, mesh.positions, np.eye(4))), \
                 patch.object(benchmark, 'build_bank', return_value=refs), \
                 patch.object(ob, 'RESULT_CAP', 6 * ob.UNIT):
                with self.assertRaises(ob.CompactBudgetExceeded):
                    benchmark.run_inference(cache, output)
            self.assertFalse((output / 'inference-seal.json').exists())
            self.assertFalse((output / 'object-000001/natural.json').exists())
            self.assertFalse((output / 'inference-progress.json').exists())
            self.assertTrue((output / 'object-000001/tracking-asset.npz.part').exists())

    def test_evaluate_and_gallery_below_floor_have_no_completion_markers(self):
        from . import prototype_benchmark as benchmark, prototype_gallery as gallery, prototype_output_budget as ob
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, result = root / 'cache', root / 'results'
            cache.mkdir()
            result.mkdir()
            storage = {'compact': 'control'}
            freeze = dict(resourceStorage=storage)
            (result / 'freeze.json').write_text(json.dumps(freeze))
            (result / 'resource-access.json').write_text('[]')
            evaluator = dict(resourceStorage=storage, dataset='BOP HB val_primesense', objects=[], benchmarkId='control', selectionDigest='a')
            (cache / 'evaluator.json').write_text(json.dumps(evaluator))
            seal = dict(resourceStorage=storage, dataset='BOP HB val_primesense', benchmarkId='control', selectionDigest='a',
                geometryBindings=[], objects=[], trialHashes={}, freezeSHA256=sha_file(result / 'freeze.json'),
                resourceAccessSHA256=sha_file(result / 'resource-access.json'), evaluatorManifestSHA256=sha_file(cache / 'evaluator.json'))
            (result / 'inference-seal.json').write_text(json.dumps(seal))
            reader = object.__new__(ResourceStore)
            def guard_factory(cache_path, output):
                return ob.CompactOutputBudget(cache_path, output, sprint_root=root, free_probe=lambda: ob.FREE_FLOOR - 1)
            with patch.object(benchmark, 'provider', return_value=reader), \
                 patch.object(benchmark, 'CompactOutputBudget', side_effect=guard_factory):
                with self.assertRaises(ob.CompactBudgetExceeded):
                    benchmark.run_evaluation(cache, result)
            self.assertFalse((result / 'report.json').exists())
            with patch.object(gallery, '_load_sealed_inputs', return_value={'seal': seal}), \
                 patch('bench.prototype_resource_store.provider', return_value=reader), \
                 patch.object(gallery, 'CompactOutputBudget', side_effect=guard_factory):
                with self.assertRaises(ob.CompactBudgetExceeded):
                    gallery.generate(cache, result)
            self.assertFalse((result / 'demo').exists())


if __name__ == '__main__':
    unittest.main()
