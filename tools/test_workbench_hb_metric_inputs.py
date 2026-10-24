"""Independent manufactured fixtures and checks for HB metric-input preparation.

These fixtures contain no downloaded HB pixels, models or annotations. Their
source identities are local test expectations, never upstream authentication.
"""
from pathlib import Path
import binascii
import copy
import hashlib
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / '.cache/video-workbench/scratch'
REVISION = '985ebaef7b1bd4bffb2039f18dd81faa647fd060'
SELECTION = 'af9d9980659b76213e1182c79551d0128dadb8b5d055650fc9b55cc8d6c86bb5'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def manufactured_ply(*, face='3 0 1 2', first_position='0 0 1000', colour='255 0 0 255'):
    """Three vertices in millimetres; exact expected output is a 10 cm triangle.

    The source is asymmetric in position and colour so an axis swap, a second
    scale conversion, index reordering or a colour-channel reversal is visible.
    """
    return ('ply\nformat ascii 1.0\ncomment manufactured, not HB data\n'
            'element vertex 3\nproperty float x\nproperty float y\nproperty float z\n'
            'property float nx\nproperty float ny\nproperty float nz\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n'
            'element face 1\nproperty list uchar int vertex_indices\nend_header\n'
            f'{first_position} 0 0 1 {colour}\n'
            '100 0 1000 0 0 1 0 255 0 255\n'
            '0 100 1000 0 0 1 0 0 255 255\n'
            f'{face}\n').encode('ascii')


EXPECTED_VERTICES_M = np.array([[0, 0, 1], [.1, 0, 1], [0, .1, 1]], dtype=np.float32)
EXPECTED_FACES = np.array([[0, 1, 2]], dtype=np.uint32)
EXPECTED_COLOURS_U8 = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)


def raw_deflate(raw):
    compressor = zlib.compressobj(level=6, wbits=-15)
    return compressor.compress(raw) + compressor.flush()


def payload_row(root, archive, member, role, raw, *, method=8):
    """A real raw-DEFLATE/stored member with independent sizes, hashes and CRC."""
    payload = raw_deflate(raw) if method == 8 else raw
    relative = 'payloads/' + digest((archive + ':' + member).encode()) + '.bin'
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {'datasetRevision': REVISION, 'archive': archive, 'member': member, 'role': role,
            'sourceMember': member, 'allowedRoles': [role], 'sourceFlags': 0, 'sourceOffset': 0,
            'method': method, 'sourceMethod': method, 'flags': 0,
            'compressedBytes': len(payload), 'expandedBytes': len(raw),
            'sourceCompressedBytes': len(payload), 'crc32': binascii.crc32(raw) & 0xffffffff,
            'compressedSHA256': digest(payload), 'uncompressedSHA256': digest(raw),
            'location': {'kind': 'payload', 'path': relative}}


def glb_document_and_binary(raw):
    """Independent GLB container reader for checking output, not its producer."""
    magic, version, size = struct.unpack_from('<III', raw)
    if (magic, version, size) != (0x46546c67, 2, len(raw)):
        raise AssertionError('Invalid manufactured-test GLB output')
    offset, document, binary = 12, None, None
    while offset < len(raw):
        length, kind = struct.unpack_from('<II', raw, offset)
        offset += 8
        chunk = raw[offset:offset + length]
        if len(chunk) != length:
            raise AssertionError('Truncated GLB output')
        if kind == 0x4e4f534a:
            document = json.loads(chunk)
        elif kind == 0x004e4942:
            binary = chunk
        offset += length
    if offset != len(raw) or document is None or binary is None:
        raise AssertionError('Incomplete GLB output')
    return document, binary


def glb_accessor(document, binary, accessor_index):
    """Read packed numeric accessor bytes without calling the conversion code."""
    accessor = document['accessors'][accessor_index]
    view = document['bufferViews'][accessor['bufferView']]
    if 'byteStride' in view:
        raise AssertionError('Extend independent test reader for interleaved output')
    dtype = {5121: np.dtype('u1'), 5123: np.dtype('<u2'), 5125: np.dtype('<u4'), 5126: np.dtype('<f4')}[accessor['componentType']]
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[accessor['type']]
    start = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    values = np.frombuffer(binary, dtype=dtype, count=accessor['count'] * width, offset=start)
    return values.reshape(accessor['count'], width).copy()


def manufactured_catalog(root):
    """Small synthetic acquisition, with a deliberately unopened annotation row.

    Tests patch the historical-anchor constants and fixed RGB byte totals to
    these manufactured expectations. This never validates the real HB inputs.
    """
    from bench.vision import cv2
    dataset = root / 'dataset'
    dataset.mkdir()
    resources = {}
    for index in range(120):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :, 0] = index * 2
        rgb[:, :, 1] = 37
        rgb[:, :, 2] = 255 - index
        ok, encoded = cv2.imencode('.png', np.ascontiguousarray(rgb[:, :, ::-1]))
        if not ok:
            raise AssertionError('Manufactured PNG encoding failed')
        member = f'val_primesense/000002/rgb/{index:06d}.png'
        resources[member] = payload_row(dataset, 'hb_val_primesense.zip', member, 'rgb', encoded.tobytes())
    model_member = 'models/obj_000001.ply'
    model = payload_row(dataset, 'hb_models.zip', model_member, 'model', manufactured_ply())
    resources[model_member] = model
    # The filter must copy neither this entry nor its path, and must not open it.
    resources['val_primesense/000002/scene_gt.json'] = {
        'location': {'kind': 'reuse', 'path': 'never-opened/scene_gt.json'}, 'allowedRoles': ['evaluation']}
    source_hashes = {'synthetic-acquisition.py': 'd' * 64}
    index = {'schemaVersion': 1, 'corpus': 'HB', 'datasetRevision': REVISION,
             'selectionDigest': SELECTION, 'storageKind': 'original_zip_members',
             'resources': resources, 'acquisitionSourceHashes': source_hashes,
             'selectedMeshBindings': [{'objectId': 1, 'meshMember': model_member,
                'meshSHA256': model['uncompressedSHA256'], 'meshExpandedBytes': model['expandedBytes'],
                'meshCRC32': f"{model['crc32']:08x}"}]}
    index_raw = json_bytes(index)
    receipt = {'status': 'complete', 'datasetRevision': REVISION, 'selectionDigest': SELECTION,
               'indexSHA256': digest(index_raw), 'sourceHashes': source_hashes}
    integrity = {'status': 'verified', 'datasetRevision': REVISION, 'selectionDigest': SELECTION,
                 'indexSHA256': digest(index_raw), 'verifiedMembers': len(resources)}
    documents = {'acquisition_receipt': ('acquisition-receipt.json', receipt),
                 'resources_index': ('resources.json', index),
                 'integrity_record': ('hb-integrity-v1.json', integrity)}
    paths, anchors = {}, {}
    for key, (name, document) in documents.items():
        path = dataset / name
        path.write_bytes(json_bytes(document))
        paths[key], anchors[key] = path, digest(path.read_bytes())
    rgb_rows = [resources[f'val_primesense/000002/rgb/{i:06d}.png'] for i in range(120)]
    return {'dataset': dataset, 'paths': paths, 'anchors': anchors, 'members': len(resources),
            'totals': {'encoded_bytes': sum(row['compressedBytes'] for row in rgb_rows),
                       'expanded_bytes': sum(row['expandedBytes'] for row in rgb_rows)}}


class HBGeometryAndPayloadTests(unittest.TestCase):
    def setUp(self):
        from tools import workbench_hb_metric_inputs as inputs
        self.inputs = inputs
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='hb-input-unit-', dir=SCRATCH)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def parse(self, raw, chunk=7):
        parser = self.inputs.AsciiPlyStream()
        for start in range(0, len(raw), chunk):
            parser.feed(raw[start:start + chunk])
        return parser.finish()

    def normalized_payload(self, raw=b'manufactured payload', method=8):
        member = 'models/obj_000001.ply'
        row = payload_row(self.root, 'hb_models.zip', member, 'model', raw, method=method)
        checked = self.inputs.validate_index_row(row, member, 'hb_models.zip', 'model', 64 << 20)
        return self.root / row['location']['path'], checked

    def test_streaming_chunk_boundaries_preserve_all_source_geometry(self):
        for chunk in (1, 7, 63, 1 << 20):
            with self.subTest(chunk=chunk):
                mesh = self.parse(manufactured_ply(), chunk)
                np.testing.assert_array_equal(mesh['positions_mm'], [[0, 0, 1000], [100, 0, 1000], [0, 100, 1000]])
                np.testing.assert_array_equal(mesh['triangles'], EXPECTED_FACES)
                np.testing.assert_array_equal(mesh['colours'], EXPECTED_COLOURS_U8)

    def test_independent_glb_bytes_preserve_mm_scale_basis_winding_and_colour(self):
        mesh = self.parse(manufactured_ply())
        raw, info = self.inputs.build_glb(mesh, name='manufactured', source_sha256='a' * 64)
        doc, binary = glb_document_and_binary(raw)
        primitive = doc['meshes'][0]['primitives'][0]
        attrs = primitive['attributes']
        np.testing.assert_array_equal(glb_accessor(doc, binary, attrs['POSITION']), EXPECTED_VERTICES_M)
        np.testing.assert_array_equal(glb_accessor(doc, binary, primitive['indices']).reshape(-1, 3), EXPECTED_FACES)
        np.testing.assert_array_equal(glb_accessor(doc, binary, attrs['COLOR_0']), EXPECTED_COLOURS_U8)
        self.assertTrue(doc['accessors'][attrs['COLOR_0']]['normalized'])
        self.assertNotIn('TEXCOORD_0', attrs)
        self.assertNotIn('images', doc)
        self.assertEqual(doc['materials'][0]['pbrMetallicRoughness']['baseColorFactor'], [1, 1, 1, 1])
        self.assertIn('KHR_materials_unlit', doc['materials'][0]['extensions'])

    def test_glb_conversion_does_not_mutate_mm_source_or_apply_scale_twice(self):
        mesh = self.parse(manufactured_ply())
        original = mesh['positions_mm'].copy()
        first, _ = self.inputs.build_glb(mesh, name='repeat', source_sha256='a' * 64)
        second, _ = self.inputs.build_glb(mesh, name='repeat', source_sha256='a' * 64)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(mesh['positions_mm'], original)

    def test_normal_conversion_emits_finite_unit_normals_or_refuses_overflow(self):
        raw = manufactured_ply().replace(b'0 0 1 255 0 0', b'0 0 1e308 255 0 0', 1)
        try:
            mesh = self.parse(raw)
            glb, _ = self.inputs.build_glb(mesh, name='large-normal', source_sha256='a' * 64)
        except self.inputs.InputsError:
            return
        doc, binary = glb_document_and_binary(glb)
        attrs = doc['meshes'][0]['primitives'][0]['attributes']
        if 'NORMAL' not in attrs:
            self.fail('Finite nonzero source normals silently omitted instead of normalized or refused')
        normals = glb_accessor(doc, binary, attrs['NORMAL'])
        self.assertTrue(np.isfinite(normals).all())
        np.testing.assert_allclose(np.linalg.norm(normals.astype(float), axis=1), 1, atol=1e-6)

    def test_finite_source_positions_that_overflow_float32_are_rejected(self):
        mesh = self.parse(manufactured_ply(first_position='1e308 0 1000'))
        with np.errstate(over='ignore', invalid='ignore'):
            with self.assertRaises(self.inputs.InputsError):
                self.inputs.build_glb(mesh, name='overflow-position', source_sha256='a' * 64)

    def test_invalid_face_topology_is_rejected(self):
        for face in ('4 0 1 2 0', '3 0 1 3', '3 -1 1 2', '3 0.5 1 2'):
            with self.subTest(face=face), self.assertRaises(self.inputs.InputsError):
                self.parse(manufactured_ply(face=face))

    def test_nonfinite_vertices_and_invalid_colours_are_rejected(self):
        for change in ({'first_position': 'nan 0 1000'}, {'first_position': 'inf 0 1000'},
                       {'colour': '256 0 0 255'}, {'colour': '0.5 0 0 255'}):
            with self.subTest(change=change), self.assertRaises(self.inputs.InputsError):
                self.parse(manufactured_ply(**change))

    def test_truncated_and_extra_body_are_rejected(self):
        original = manufactured_ply()
        for raw in (original.rsplit(b'3 0 1 2', 1)[0], original + b'3 0 1 2\n'):
            with self.subTest(raw=raw[-30:]), self.assertRaises(self.inputs.InputsError):
                self.parse(raw)

    def test_execution_face_cap_is_not_the_portability_target(self):
        raw = manufactured_ply().replace(b'element face 1', b'element face 200001')
        with self.assertRaises(self.inputs.InputsError):
            self.parse(raw)
        self.assertEqual(self.inputs.FACE_CAP, 200000)
        self.assertEqual(self.inputs.PORTABLE_TRIANGLES, 20000)

    def test_non_mm_header_units_are_refused(self):
        raw = manufactured_ply().replace(b'element vertex', b'comment units: metres\nelement vertex', 1)
        with self.assertRaises(self.inputs.InputsError):
            self.parse(raw)

    def test_raw_deflate_and_stored_payloads_match_independent_hash_crc_and_bytes(self):
        raw = manufactured_ply()
        for method in (0, 8):
            with self.subTest(method=method):
                path, row = self.normalized_payload(raw, method)
                parts = []
                result = self.inputs.verified_payload_stream(path, row, parts.append)
                self.assertEqual(b''.join(parts), raw)
                self.assertEqual(result['uncompressed_sha256'], digest(raw))
                self.assertEqual(result['crc32'], binascii.crc32(raw) & 0xffffffff)

    def test_sha_crc_or_length_mismatch_never_passes_payload_integrity(self):
        path, row = self.normalized_payload()
        for field, value in [('compressed_sha256', '0' * 64), ('uncompressed_sha256', '0' * 64),
                             ('crc32', row['crc32'] ^ 1), ('compressed_bytes', row['compressed_bytes'] + 1)]:
            with self.subTest(field=field), self.assertRaises(self.inputs.InputsError):
                self.inputs.verified_payload_stream(path, {**row, field: value})

    def test_trailing_deflate_bytes_rejected_even_with_updated_compressed_hash(self):
        path, row = self.normalized_payload()
        corrupt = path.read_bytes() + b'trailing'
        path.write_bytes(corrupt)
        row.update(compressed_bytes=len(corrupt), compressed_sha256=digest(corrupt))
        with self.assertRaisesRegex(self.inputs.InputsError, 'trailing|DEFLATE'):
            self.inputs.verified_payload_stream(path, row)

    def test_truncated_deflate_rejected_even_with_updated_compressed_hash(self):
        path, row = self.normalized_payload()
        corrupt = path.read_bytes()[:-1]
        path.write_bytes(corrupt)
        row.update(compressed_bytes=len(corrupt), compressed_sha256=digest(corrupt))
        with self.assertRaisesRegex(self.inputs.InputsError, 'truncated|DEFLATE'):
            self.inputs.verified_payload_stream(path, row)

    def test_overexpansion_stops_before_forwarding_oversized_data(self):
        path, row = self.normalized_payload(b'x' * 20000)
        parts = []
        with self.assertRaisesRegex(self.inputs.InputsError, 'expansion'):
            self.inputs.verified_payload_stream(path, {**row, 'expanded_bytes': 1}, parts.append)
        self.assertLessEqual(sum(map(len, parts)), 1)

    def test_strict_metadata_parser_rejects_numeric_overflow_not_only_nan_tokens(self):
        for raw in (b'{"value":1e400}', b'{"value":-1e400}', b'{"value":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(self.inputs.InputsError):
                self.inputs.strict_json(raw, 'manufactured metadata')


class HBPreparationStageTests(unittest.TestCase):
    def setUp(self):
        from tools import workbench_hb_metric_inputs as inputs
        self.inputs = inputs
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='hb-stage-unit-', dir=SCRATCH)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.workspace = self.root / 'workspace'
        (self.workspace / 'derived').mkdir(parents=True)
        self.ctx = inputs.Context(root=self.root, workspace=self.workspace)
        self.catalog = manufactured_catalog(self.root)
        for name, value in [('ANCHORS', self.catalog['anchors']), ('INTEGRITY_MEMBERS', self.catalog['members']),
                            ('RGB_TOTALS', self.catalog['totals'])]:
            patcher = mock.patch.object(inputs, name, value)
            patcher.start(); self.addCleanup(patcher.stop)

    def filter(self, name='allowlist-output'):
        paths, anchors = self.catalog['paths'], self.catalog['anchors']
        return self.inputs.filter_allowlist(self.ctx,
            receipt_path=str(paths['acquisition_receipt']), expected_receipt_sha256=anchors['acquisition_receipt'],
            index_path=str(paths['resources_index']), expected_index_sha256=anchors['resources_index'],
            integrity_path=str(paths['integrity_record']), expected_integrity_sha256=anchors['integrity_record'],
            object_ids=[1], out=str(self.ctx.derived / name))

    def allowlist(self):
        result = self.filter()
        return self.root / result['allowlist'], result['allowlist_sha256']

    def test_real_guarded_filter_opens_no_payloads_or_annotation_paths(self):
        result = self.filter()
        doc = json.loads((self.root / result['allowlist']).read_bytes())
        self.assertEqual(result['payload_opens'], 0)
        self.assertEqual(len(doc['rgb']), 120)
        self.assertEqual(len(doc['models']), 1)
        self.assertFalse(doc['trust']['upstream_authenticated'])
        self.assertNotIn('scene_gt.json', json.dumps(doc))
        self.assertNotIn('never-opened', json.dumps(doc))

    def test_missing_or_forged_historical_anchor_rejected_before_payload_opens(self):
        self.catalog['paths']['resources_index'].write_bytes(b'{}')
        with mock.patch.object(self.inputs, 'verified_payload_stream', side_effect=AssertionError('payload opened')):
            with self.assertRaises(self.inputs.InputsError):
                self.filter()
        self.assertFalse((self.ctx.derived / 'allowlist-output').exists())

    def test_validated_model_stage_roundtrips_real_glb_with_guard_and_real_disk_budget(self):
        allow, expected = self.allowlist()
        result = self.inputs.prepare_model(self.ctx, allowlist=str(allow), expected_allowlist_sha256=expected,
            object_id=1, existing_input_bytes=0, out=str(self.ctx.derived / 'model-output'))
        receipt = json.loads((self.root / result['receipt']).read_bytes())
        self.assertTrue(receipt['closure']['after_matches'])
        self.assertFalse(receipt['mobile_verified'])
        self.assertEqual(receipt['units']['factor'], .001)
        self.assertTrue(receipt['geometry']['roundtrip']['producer_glb_mesh_positions_equal'])
        self.assertEqual(receipt['geometry']['triangles'], 1)

    def test_explicit_preparation_scope_works_and_cannot_be_rebound_to_another_control(self):
        scope = self.ctx.derived / 'explicit-control'; scope.mkdir()
        self.ctx = self.inputs.Context(root=self.root, workspace=self.workspace, preparation=scope)
        result = self.filter(name='explicit-control/allowlist-output')
        allow = self.root / result['allowlist']; expected = result['allowlist_sha256']
        prepared = self.inputs.prepare_model(self.ctx, allowlist=str(allow), expected_allowlist_sha256=expected,
            object_id=1, existing_input_bytes=0, out=str(scope / 'model-output'))
        self.assertTrue((self.root / prepared['receipt']).is_file())
        other = self.ctx.derived / 'another-control'; other.mkdir()
        other_ctx = self.inputs.Context(root=self.root, workspace=self.workspace, preparation=other)
        with mock.patch.object(self.inputs, 'verified_payload_stream', side_effect=AssertionError('payload opened')):
            with self.assertRaises(self.inputs.InputsError):
                self.inputs.prepare_model(other_ctx, allowlist=str(allow), expected_allowlist_sha256=expected,
                    object_id=1, existing_input_bytes=0, out=str(other / 'model-output'))
        self.assertFalse((other / 'model-output').exists())
        with self.assertRaises(self.inputs.InputsError):
            self.inputs.prepare_model(self.ctx, allowlist=str(allow), expected_allowlist_sha256=expected,
                object_id=1, existing_input_bytes=0, out=str(self.ctx.derived / 'outside-explicit-scope'))

    def test_preparation_refuses_low_disk_before_payload_decompression(self):
        allow, expected = self.allowlist()
        with mock.patch.object(self.inputs.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 0})()), \
             mock.patch.object(self.inputs, 'verified_payload_stream', side_effect=AssertionError('payload opened')):
            with self.assertRaises(self.inputs.Blocked):
                self.inputs.prepare_model(self.ctx, allowlist=str(allow), expected_allowlist_sha256=expected,
                    object_id=1, existing_input_bytes=0, out=str(self.ctx.derived / 'low-disk-output'))
        self.assertFalse((self.ctx.derived / 'low-disk-output').exists())

    def test_model_stage_failure_keeps_failure_receipt_without_success_receipt(self):
        allow, expected = self.allowlist()
        out = self.ctx.derived / 'failed-model-output'
        with mock.patch.object(self.inputs, 'build_glb', side_effect=self.inputs.InputsError('manufactured failure')):
            with self.assertRaises(self.inputs.InputsError):
                self.inputs.prepare_model(self.ctx, allowlist=str(allow), expected_allowlist_sha256=expected,
                    object_id=1, existing_input_bytes=0, out=str(out))
        self.assertTrue((out / 'failure.json').is_file())
        self.assertFalse((out / 'model-receipt.json').exists())

    def test_unselected_payload_and_gt_opens_are_blocked_in_any_mode_and_guard_resets(self):
        gt = self.root / 'scene_gt.json'
        gt.write_bytes(b'{"9":[]}')
        payload = self.root / 'unselected' / 'payloads' / 'other.bin'
        payload.parent.mkdir(parents=True); payload.write_bytes(b'synthetic')
        with self.inputs.guarded_stage(self.ctx):
            for target in (gt, payload):
                for mode in ('rb', 'r+b', 'wb'):
                    with self.subTest(target=target.name, mode=mode), self.assertRaises(self.inputs.GuardError):
                        with target.open(mode):
                            self.fail('forbidden file opened')
        self.assertEqual(payload.read_bytes(), b'synthetic')
        self.assertEqual(gt.read_bytes(), b'{"9":[]}')

    def test_rehashed_allowlist_cannot_drop_historical_anchor_provenance(self):
        path, _ = self.allowlist()
        doc = json.loads(path.read_bytes())
        doc['anchors'] = {}
        raw = json_bytes(doc); forged = self.root / 'forged-allowlist.json'; forged.write_bytes(raw)
        with self.assertRaises(self.inputs.InputsError):
            self.inputs.load_allowlist(self.ctx, str(forged), digest(raw))

    def test_rehashed_allowlist_cannot_change_supported_payload_compression_method(self):
        path, _ = self.allowlist()
        doc = json.loads(path.read_bytes()); doc['models'][0]['method'] = 47
        raw = json_bytes(doc); forged = self.root / 'forged-method-allowlist.json'; forged.write_bytes(raw)
        with self.assertRaises(self.inputs.InputsError):
            self.inputs.load_allowlist(self.ctx, str(forged), digest(raw))

    def test_exact_metadata_read_exception_still_forbids_mutation(self):
        metadata = self.root / 'scene_gt.json'
        metadata.write_bytes(b'{"9":[]}')
        with self.inputs.guarded_stage(self.ctx, read_exceptions=[metadata]):
            self.assertEqual(metadata.read_bytes(), b'{"9":[]}')
            with self.assertRaises(self.inputs.GuardError):
                metadata.unlink()

    def test_read_exception_cannot_be_renamed_truncated_or_opened_for_update(self):
        metadata = self.root / 'scene_gt.json'
        raw = b'{"9":[]}'
        metadata.write_bytes(raw)
        with self.inputs.guarded_stage(self.ctx, read_exceptions=[metadata]):
            for action in (lambda: metadata.rename(self.root / 'renamed.json'),
                           lambda: metadata.open('r+b'), lambda: metadata.open('wb')):
                with self.assertRaises(self.inputs.GuardError):
                    action()
        self.assertEqual(metadata.read_bytes(), raw)
        self.assertFalse((self.root / 'renamed.json').exists())

    def test_guard_does_not_grant_arbitrary_subprocess_capability(self):
        with self.inputs.guarded_stage(self.ctx):
            with self.assertRaises(self.inputs.GuardError):
                subprocess.run([sys.executable, '-B', '-S', '-c', 'raise SystemExit(99)'], check=True)

    def test_publication_rechecks_disk_reserve_before_any_file_write(self):
        out = self.ctx.derived / 'publication-output'; out.mkdir()
        ledger = self.inputs.OutputLedger(self.ctx, out, 'filter-allowlist')
        with self.inputs.guarded_stage(self.ctx), \
             mock.patch.object(self.inputs.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 0})()):
            with self.assertRaises(self.inputs.Blocked):
                self.inputs.publish_bytes(ledger, 'must-not-write.json', b'{}')
        self.assertEqual(list(out.iterdir()), [])

    def test_preparation_budget_includes_retained_failed_sibling_attempts(self):
        for metadata, total_cap, metadata_cap, retained_bytes in ((False, 128, 64, 129), (True, 1024, 64, 65)):
            with self.subTest(metadata=metadata):
                parent = self.ctx.derived / ('metadata-budget' if metadata else 'total-budget')
                parent.mkdir()
                # A distinct context makes this manufactured preparation scope explicit.
                scoped_workspace = parent / 'workspace'; scoped_derived = scoped_workspace / 'derived'
                scoped_derived.mkdir(parents=True)
                ctx = self.inputs.Context(root=self.root, workspace=scoped_workspace)
                failed = scoped_derived / 'prior-failed-attempt'; failed.mkdir()
                (failed / ('failure.json' if metadata else '.retained-partial.mp4')).write_bytes(b'x' * retained_bytes)
                out = scoped_derived / 'new-stage'; out.mkdir()
                with mock.patch.object(self.inputs, 'FINAL_OUTPUTS_CAP', total_cap), \
                     mock.patch.object(self.inputs, 'METADATA_AGGREGATE_CAP', metadata_cap), \
                     self.inputs.guarded_stage(ctx):
                    with self.assertRaises(self.inputs.Blocked):
                        ledger = self.inputs.OutputLedger(ctx, out, 'filter-allowlist')
                        self.inputs.publish_bytes(ledger, 'must-not-write.json', b'{}')
                self.assertEqual(list(out.iterdir()), [])

    def test_encoder_stops_before_the_chunk_that_would_exceed_the_video_cap(self):
        # No process and no output file: test the actual drain against an in-memory sink.
        encoder = object.__new__(self.inputs.FfmpegPipeEncoder)
        encoder.proc = SimpleNamespace(stdout=io.BytesIO(b'abcdefgh'))
        encoder.written = 0; encoder.error = None; encoder.fd = 987654
        encoder.ledger = self.inputs.OutputLedger(self.ctx, self.ctx.derived, 'prepare-video')
        encoder._kill = mock.Mock()
        written = bytearray()
        def write(fd, data):
            self.assertEqual(fd, 987654)
            written.extend(data)
            return len(data)
        with mock.patch.object(self.inputs, 'VIDEO_CAP', 5), \
             mock.patch.object(self.inputs, 'PIPE_CHUNK', 4), \
             mock.patch.object(self.inputs.os, 'write', side_effect=write):
            encoder._drain_stdout()
        self.assertEqual(bytes(written), b'abcd')
        self.assertEqual(encoder.written, 4)
        self.assertIsInstance(encoder.error, self.inputs.Blocked)
        encoder._kill.assert_called_once()

    def test_encoder_does_not_accumulate_unbounded_stderr_in_memory(self):
        import collections
        encoder = object.__new__(self.inputs.FfmpegPipeEncoder)
        encoder.proc = SimpleNamespace(stderr=io.BytesIO(b'x' * (1 << 20)))
        encoder.stderr_tail = collections.deque(); encoder.stderr_bytes = 0
        encoder._drain_stderr()
        self.assertEqual(encoder.stderr_bytes, 1 << 20)
        self.assertLessEqual(sum(map(len, encoder.stderr_tail)), self.inputs.STDERR_TAIL)

    def test_encoder_retries_partial_stdin_writes_without_dropping_frame_bytes(self):
        # Raw pipe writes may legally consume fewer bytes than requested.
        # This is independent of FFmpeg and needs no subprocess or file.
        payload = bytearray()
        class ShortPipe:
            def write(self, data):
                consumed = min(65537, len(data))
                payload.extend(data[:consumed])
                return consumed
        encoder = object.__new__(self.inputs.FfmpegPipeEncoder)
        encoder.proc = SimpleNamespace(stdin=ShortPipe())
        encoder.error = None
        rgb = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        encoder.write_frame(rgb)
        self.assertEqual(bytes(payload), rgb.tobytes())

    def test_encoder_refuses_zero_or_unknown_stdin_progress(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        for progress in (0, None):
            encoder = object.__new__(self.inputs.FfmpegPipeEncoder)
            encoder.proc = SimpleNamespace(stdin=SimpleNamespace(write=lambda data: progress))
            encoder.error = None
            with self.subTest(progress=progress), self.assertRaises(self.inputs.InputsError):
                encoder.write_frame(rgb)

    def test_real_encoder_overflow_reaps_process_closes_pipes_and_keeps_no_success_receipt(self):
        allow, expected = self.allowlist()
        out = self.ctx.derived / 'overflow-video-output'
        encoders = []
        original_enter = self.inputs.FfmpegPipeEncoder.__enter__
        def enter(encoder):
            encoders.append(encoder)
            return original_enter(encoder)
        with mock.patch.object(self.inputs, 'VIDEO_CAP', 128), \
             mock.patch.object(self.inputs, 'FFMPEG_DEADLINE_S', 5), \
             mock.patch.object(self.inputs.FfmpegPipeEncoder, '__enter__', enter):
            with self.assertRaises(self.inputs.InputsError):
                self.inputs.prepare_video(self.ctx, allowlist=str(allow), expected_allowlist_sha256=expected,
                    existing_input_bytes=0, out=str(out))
        self.assertEqual(len(encoders), 1)
        encoder = encoders[0]
        self.assertIsNotNone(encoder.proc)
        self.assertIsNotNone(encoder.proc.poll())
        self.assertTrue(all(pipe.closed for pipe in (encoder.proc.stdin, encoder.proc.stdout, encoder.proc.stderr)))
        self.assertTrue(all(not thread.is_alive() for thread in encoder.threads))
        self.assertLessEqual(encoder.written, 128)
        self.assertIsNone(encoder.fd)
        self.assertFalse(self.inputs._GUARD['active'])
        self.assertFalse((out / 'video-receipt.json').exists())
        for staging in out.glob('.staging-*.mp4'):
            self.assertLessEqual(staging.stat().st_size, 128)
        self.assertTrue((out / 'failure.json').is_file())

    def test_allowlist_revalidation_rejects_rehashed_conflicting_contracts_before_payload_read(self):
        path, _ = self.allowlist()
        accepted = json.loads(path.read_bytes())
        mutations = {
            'scene': lambda d: d['source'].__setitem__('scene', '000003'),
            'archive': lambda d: d['source'].__setitem__('rgb_archive', 'different.zip'),
            'revision': lambda d: d['source'].__setitem__('dataset_revision', '0' * 40),
            'seed moved into scoring': lambda d: d['frozen_split'].__setitem__('seed_index', 10),
            'scoring leaks into preparation': lambda d: d['frozen_split']['opening'].append(10),
            'duplicate model': lambda d: d['models'].append(copy.deepcopy(d['models'][0])),
            'invalid SHA': lambda d: d['rgb'][10].__setitem__('compressed_sha256', 'not-a-hash'),
            'boolean size': lambda d: d['rgb'][10].__setitem__('compressed_bytes', True),
            'zero expanded size': lambda d: d['rgb'][10].__setitem__('expanded_bytes', 0),
            'negative offset': lambda d: d['rgb'][10].__setitem__('source_offset', -1),
            'encrypted payload': lambda d: d['rgb'][10].__setitem__('flags', 1),
            'overflow CRC': lambda d: d['rgb'][10].__setitem__('crc32', 1 << 32),
            'wrong role': lambda d: d['rgb'][10].__setitem__('roles', ['evaluation']),
            'wrong location': lambda d: d['rgb'][10].__setitem__('payload_path', '../scene_gt.json'),
            'wrong split': lambda d: d['rgb'][9].__setitem__('split', 'scoring'),
            'wrong aggregate': lambda d: d['rgb_totals'].__setitem__('encoded_bytes', 1),
            'missing source seals': lambda d: d.pop('source_seals'),
            'unverified anchor': lambda d: d['anchors']['resources_index'].__setitem__('verified_before_parsing', False),
        }
        for name, mutate in mutations.items():
            forged = copy.deepcopy(accepted); mutate(forged)
            raw = json_bytes(forged)
            target = self.root / 'forged-contract.json'; target.write_bytes(raw)
            with self.subTest(name=name), \
                 mock.patch.object(self.inputs, 'verified_payload_stream', side_effect=AssertionError('payload opened')), \
                 self.assertRaises(self.inputs.InputsError):
                self.inputs.prepare_model(self.ctx, allowlist=str(target), expected_allowlist_sha256=digest(raw),
                    object_id=1, existing_input_bytes=0, out=str(self.ctx.derived / 'must-not-exist'))
        self.assertFalse((self.ctx.derived / 'must-not-exist').exists())

    def selection_fixture(self, allow_path):
        allow = json.loads(allow_path.read_bytes())
        row = allow['rgb'][9]
        review = {'schema': self.inputs.SCHEMA_SELECTION_REVIEW, 'scene': '000002',
                  'frame_index': 9, 'frame_id': 's000009', 'rgb_member': row['member'],
                  'rgb_source_sha256': row['uncompressed_sha256'], 'object_id': 1, 'instance_index': 0,
                  'positive': [[320, 240]], 'negative': [[5, 5]], 'reviewed': True,
                  'reviewed_by': 'manufactured fixture; no real object claim',
                  'attestation': 'Synthetic opening-only selection, not a real HB binding.',
                  'opening_only': True, 'derived_from_scored_frames': False}
        path = self.root / 'selection-review.json'; path.write_bytes(json_bytes(review))
        return path, digest(path.read_bytes())

    def seed_fixture(self, allow_path, allow_sha, *, mutate_during_scan=None, mutate_during_publication=None):
        from tools import workbench_hb_opening_seed as seed
        from tools.test_workbench_hb_opening_seed import IDENTITY_R, manufactured_scene
        review_path, review_sha = self.selection_fixture(allow_path)
        raw = manufactured_scene(selected=[{'obj_id': 1, 'cam_R_m2c': IDENTITY_R,
                                             'cam_t_m2c': [100, 200, 1000]}])
        gt = self.root / 'scene_gt.json'; gt.write_bytes(raw)
        original = seed.extract_frame_value
        original_publish = self.inputs.publish_bytes
        def scanning(chunks, *args, **kwargs):
            value = original(chunks, *args, **kwargs)
            if mutate_during_scan:
                mutate_during_scan(allow_path, review_path)
            return value
        def publishing(*args, **kwargs):
            result = original_publish(*args, **kwargs)
            if len(args) > 1 and args[1] == 'seed.json' and mutate_during_publication:
                mutate_during_publication(allow_path, review_path)
            return result
        with mock.patch.object(seed, 'extract_frame_value', side_effect=scanning), \
             mock.patch.object(self.inputs, 'publish_bytes', side_effect=publishing):
            result = seed.extract_seed(self.ctx, scene_gt=str(gt), expected_scene_gt_sha256=digest(raw),
                frame_index=9, object_id=1, instance_index=0, selection_review=str(review_path),
                expected_selection_review_sha256=review_sha, allowlist=str(allow_path),
                expected_allowlist_sha256=allow_sha, out=str(self.ctx.derived / 'seed-output'))
        return result, review_path

    def test_full_seed_stage_extracts_only_selected_pose_and_never_reads_any_payload(self):
        from tools.test_workbench_hb_opening_seed import EXPECTED_SEED_M
        allow, expected = self.allowlist()
        with mock.patch.object(self.inputs, 'verified_payload_stream', side_effect=AssertionError('payload opened')):
            result, review = self.seed_fixture(allow, expected)
        doc = json.loads((self.root / result['seed']).read_bytes())
        np.testing.assert_array_equal(doc['seed_pose_m'], EXPECTED_SEED_M)
        self.assertEqual(doc['allowlist_sha256'], expected)
        self.assertEqual(doc['selection_review_sha256'], digest(review.read_bytes()))
        self.assertEqual(doc['applies_once_to_frame_id'], 's000010')
        self.assertIs(doc['reuse_on_recovery'], False)
        self.assertIs(doc['other_frames_materialized'], False)
        self.assertNotIn('NEVER_MATERIALIZE_FUTURE', json.dumps(doc))

    def test_seed_stage_cannot_bind_changed_allowlist_bytes_after_validating_original(self):
        allow, expected = self.allowlist()
        def mutate(path, review):
            # Whitespace preserves JSON meaning but must invalidate the byte identity.
            path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaises(self.inputs.InputsError):
            self.seed_fixture(allow, expected, mutate_during_scan=mutate)
        self.assertFalse((self.ctx.derived / 'seed-output' / 'seed.json').exists())

    def test_seed_stage_cannot_bind_changed_selection_review_after_validating_original(self):
        allow, expected = self.allowlist()
        def mutate(path, review):
            review.write_bytes(review.read_bytes() + b'\n')
        with self.assertRaises(self.inputs.InputsError):
            self.seed_fixture(allow, expected, mutate_during_scan=mutate)
        self.assertFalse((self.ctx.derived / 'seed-output' / 'seed.json').exists())

    def test_seed_publication_rechecks_inputs_before_publishing_its_success_receipt(self):
        allow, expected = self.allowlist()
        def mutate(path, review):
            review.write_bytes(review.read_bytes() + b'\n')
        with self.assertRaises(self.inputs.InputsError):
            self.seed_fixture(allow, expected, mutate_during_publication=mutate)
        self.assertFalse((self.ctx.derived / 'seed-output' / 'seed-receipt.json').exists())

    def test_filter_rejects_source_changes_before_success_publication(self):
        original = self.inputs.source_closure
        calls = []
        def changing_sources():
            value = original()
            calls.append(1)
            if len(calls) > 1:
                value = dict(value); value['manufactured-change.py'] = 'f' * 64
            return value
        with mock.patch.object(self.inputs, 'source_closure', side_effect=changing_sources):
            with self.assertRaises(self.inputs.InputsError):
                self.filter()
        self.assertFalse((self.ctx.derived / 'allowlist-output' / 'allowlist.json').exists())

    def test_colourless_model_cannot_be_reported_as_prepared_vertex_colour_control(self):
        allow, expected = self.allowlist()
        original = self.inputs.AsciiPlyStream.finish
        def missing_colour(parser):
            mesh = original(parser); mesh['colours'] = None
            return mesh
        with mock.patch.object(self.inputs.AsciiPlyStream, 'finish', missing_colour):
            with self.assertRaises(self.inputs.InputsError):
                self.inputs.prepare_model(self.ctx, allowlist=str(allow), expected_allowlist_sha256=expected,
                    object_id=1, existing_input_bytes=0, out=str(self.ctx.derived / 'no-colour-output'))
        self.assertFalse((self.ctx.derived / 'no-colour-output' / 'model-receipt.json').exists())

    def test_manufactured_video_and_complete_assembly_reject_contradictory_rehashed_receipts(self):
        """Exercise the real CPU codec, guard, model producer, seed and core loaders.

        No neural modules or actual HB inputs are used; all scratch is temporary.
        Rehashed receipts must not override independently checked media contracts.
        """
        from bench import workbench_contract as wc
        from bench import workbench_neural as wn
        from bench.vision import cv2
        allow, allow_sha = self.allowlist()
        model_result = self.inputs.prepare_model(self.ctx, allowlist=str(allow),
            expected_allowlist_sha256=allow_sha, object_id=1, existing_input_bytes=0,
            out=str(self.ctx.derived / 'model-output'))
        video_result = self.inputs.prepare_video(self.ctx, allowlist=str(allow),
            expected_allowlist_sha256=allow_sha, existing_input_bytes=0,
            out=str(self.ctx.derived / 'video-output'))
        model_path = self.root / model_result['receipt']
        video_path = self.root / video_result['receipt']
        model = json.loads(model_path.read_bytes()); video = json.loads(video_path.read_bytes())
        self.assertEqual(len(video['frames']), 120)
        self.assertEqual(video['decode']['sequential_frames'], 120)
        self.assertIsNone(video['clock']['physical_fps'])
        self.assertIsNone(video['clock']['physical_timestamps'])
        self.assertEqual(video['clock']['presentation_fps'], 30)
        self.assertEqual(video['video']['bytes'], (self.root / video['video']['path']).stat().st_size)
        self.assertLessEqual(video['video']['bytes'], 6 << 20)
        allow_doc = json.loads(allow.read_bytes())
        for index, frame in enumerate(video['frames']):
            self.assertEqual((frame['index'], frame['frame_id']), (index, f's{index:06d}'))
            self.assertEqual(frame['png_sha256'], allow_doc['rgb'][index]['uncompressed_sha256'])
        # Independent decoder check, including EOF and coarse lossy colour/order.
        cap = cv2.VideoCapture(str(self.root / video['video']['path']))
        self.assertTrue(cap.isOpened())
        decoded = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                self.assertEqual(bgr.shape, (480, 640, 3))
                rgb_mean = bgr[:, :, ::-1].mean(axis=(0, 1))
                np.testing.assert_allclose(rgb_mean, [decoded * 2, 37, 255 - decoded], atol=8)
                decoded += 1
        finally:
            cap.release()
        self.assertEqual(decoded, 120)
        seed_result, review_path = self.seed_fixture(allow, allow_sha)
        seed_path = self.root / seed_result['seed']
        calibration_path = self.root / 'calibrations.json'
        calibration_path.write_bytes(json_bytes({'schemaVersion': 1, 'calibrations': [
            {'kind': 'dataset_reference', 'width': 640, 'height': 480, 'K': list(self.inputs.EXPECTED_K)}]}))
        placement = {'schema': self.inputs.SCHEMA_PLACEMENT_REVIEW,
            'seed_sha256': digest(seed_path.read_bytes()), 'glb_sha256': model['glb']['sha256'],
            'video_sha256': video['video']['sha256'], 'selection_review_sha256': digest(review_path.read_bytes()),
            'K': list(self.inputs.EXPECTED_K), 'principal_point_origin': 'integer_centers',
            'principal_point_evidence': 'reviewed_assumption', 'anchor_frame_id': 's000009',
            'anchor_pixel': [320, 240], 'qualitative_review': 'Manufactured fixture, not physical proof.',
            'reviewed_by': 'Codex synthetic contract test', 'physical_proof': False}
        placement_path = self.root / 'placement-review.json'; placement_path.write_bytes(json_bytes(placement))
        args = dict(allowlist=str(allow), expected_allowlist_sha256=allow_sha,
            model_receipt=str(model_path), expected_model_receipt_sha256=digest(model_path.read_bytes()),
            video_receipt=str(video_path), expected_video_receipt_sha256=digest(video_path.read_bytes()),
            seed=str(seed_path), expected_seed_sha256=digest(seed_path.read_bytes()),
            selection_review=str(review_path), expected_selection_review_sha256=digest(review_path.read_bytes()),
            placement_review=str(placement_path), expected_placement_review_sha256=digest(placement_path.read_bytes()),
            calibrations=str(calibration_path), expected_calibrations_sha256=digest(calibration_path.read_bytes()))
        result = self.inputs.assemble(self.ctx, **args, out=str(self.ctx.derived / 'assembly-output'))
        self.assertEqual(result['status'], 'assembled_unadmitted_unregistered')
        bundle = wn.MetricBundle.from_file(self.ctx.derived / 'assembly-output/metric-bundle.json', root=self.root)
        self.assertEqual(bundle.family, 'hb_primesense_640x480')
        self.assertEqual(bundle.seed_frame_id, 's000009')
        clip = json.loads((self.ctx.derived / 'assembly-output/clip-proposal.json').read_bytes())
        wc.ClipSpec.from_dict(clip)
        self.assertFalse(clip['provenance']['admitted'])
        self.assertFalse(clip['provenance']['registered'])
        self.assertEqual(len(clip['scoring_frame_ids']), 110)
        contradictions = [
            ('model', 'different source', lambda d: d['source'].__setitem__('uncompressed_sha256', 'f' * 64)),
            ('model', 'wrong units', lambda d: d['units'].__setitem__('factor', 1)),
            ('model', 'false size', lambda d: d['glb'].__setitem__('bytes', 1)),
            ('model', 'too many triangles', lambda d: d['geometry'].__setitem__('triangles', 200001)),
            ('model', 'wrong vertex count', lambda d: d['geometry'].__setitem__('vertices', 1)),
            ('model', 'failed position roundtrip', lambda d: d['geometry']['roundtrip'].__setitem__('positions_bitwise_equal_float32', False)),
            ('model', 'failed winding roundtrip', lambda d: d['geometry']['roundtrip'].__setitem__('indices_and_winding_equal', False)),
            ('model', 'failed colour roundtrip', lambda d: d['geometry']['roundtrip'].__setitem__('vertex_colours_equal', False)),
            ('model', 'failed normal roundtrip', lambda d: d['geometry']['roundtrip'].__setitem__('normals_equal', False)),
            ('model', 'wrong producer', lambda d: d.__setitem__('tool_version', 'unreviewed')),
            ('video', 'wrong source frame', lambda d: d['frames'][17].__setitem__('png_sha256', 'f' * 64)),
            ('video', 'missing last frame', lambda d: d['frames'].pop()),
            ('video', 'false decode count', lambda d: d['decode'].__setitem__('sequential_frames', 119)),
            ('video', 'invented physical FPS', lambda d: d['clock'].__setitem__('physical_fps', 30)),
            ('video', 'changed presentation FPS', lambda d: d['clock'].__setitem__('presentation_fps', 1)),
            ('video', 'false physical prefix', lambda d: d['clock'].__setitem__('prefix_attestation', '10 physical seconds verified')),
            ('video', 'declared crop', lambda d: d['geometry'].__setitem__('crop', [0, 0, 100, 100])),
            ('video', 'declared rotation', lambda d: d['geometry'].__setitem__('rotation', 90)),
            ('video', 'contradictory EOF', lambda d: d['decode'].__setitem__('end_of_video_after', 119)),
            ('video', 'failed seek claim', lambda d: d['decode']['frame_reader_seek_check'].__setitem__('exact', False)),
            ('video', 'failed process', lambda d: d['process'].__setitem__('exit_code', 7)),
            ('video', 'false stdout size', lambda d: d['process'].__setitem__('stdout_bytes', 1)),
            ('video', 'changed split', lambda d: d['split']['opening'].append(10)),
        ]
        for index, (kind, name, mutate) in enumerate(contradictions):
            document = copy.deepcopy(model if kind == 'model' else video); mutate(document)
            raw = json_bytes(document); forged = self.root / 'forged-preparation.json'; forged.write_bytes(raw)
            changed = dict(args, **{f'{kind}_receipt': str(forged),
                                  f'expected_{kind}_receipt_sha256': digest(raw)})
            target = self.ctx.derived / f'refused-assembly-{index}'
            with self.subTest(name=name):
                # Test the nested contract itself, separately from any later
                # preparation-root or receipt-path restrictions in assemble.
                with self.assertRaises(self.inputs.InputsError):
                    if kind == 'model':
                        self.inputs.validate_model_receipt(self.ctx, document, allow_doc, allow_sha,
                            json.loads(review_path.read_bytes()), self.inputs.source_closure(), self.inputs.runtime_identity())
                    else:
                        self.inputs.validate_video_receipt(self.ctx, document, allow_doc, allow_sha,
                            self.inputs.source_closure(), self.inputs.runtime_identity())
                with self.assertRaises(self.inputs.InputsError):
                    self.inputs.assemble(self.ctx, **changed, out=str(target))
                self.assertFalse((target / 'assembly-receipt.json').exists())
