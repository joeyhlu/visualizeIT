"""Lossless, bounded per-member ZIP payload storage with explicit read phases.

The index is storage provenance, never an inference label API. No shared image
cache is kept. Expanded bytes are verified before a seekable stream is returned.
"""
from __future__ import annotations

import binascii
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zlib

CHUNK = 1024 * 1024
MAX_EXPANDED = 32 * CHUNK
REUSE_ROOT = Path(__file__).resolve().parents[1] / '.cache/prototype-sprint/hb-scene-metadata-v1'
PHASE_ROLES = {
    'inference': frozenset(('model', 'model_info', 'rgb', 'onboarding_depth')),
    'evaluation': frozenset(('model', 'rgb', 'evaluation_depth', 'evaluation_mask')),
    'gallery': frozenset(('model', 'rgb', 'evaluation_depth')),
}


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(CHUNK), b''):
            value.update(chunk)
    return value.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Unsafe resource name')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Unsafe resource name')
    return path


def inside(root, relative):
    path = (Path(root) / safe_relative(relative)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Resource escapes storage root')
    return path


def valid_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid SHA256')
    return value


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('Duplicate JSON key in sealed index')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def verified_stream(path, row):
    """Verify compressed bytes, exact expansion, CRC and SHA with bounded chunks."""
    size, compressed = row['expandedBytes'], row['compressedBytes']
    method = row['method']
    if (type(size) is not int or not 0 <= size <= MAX_EXPANDED
            or type(compressed) is not int or not 0 <= compressed <= MAX_EXPANDED
            or method not in (0, 8) or Path(path).stat().st_size != compressed):
        raise ValueError('Invalid resource size or method')
    chash, uhash, crc, count = hashlib.sha256(), hashlib.sha256(), 0, 0
    output = io.BytesIO()
    if size:
        # Allocate the declared buffer once; subsequent bounded writes cannot
        # trigger geometric buffer growth beyond the declared expanded extent.
        output.seek(size - 1)
        output.write(b'\0')
        output.seek(0)
    decoder = zlib.decompressobj(-15) if method == 8 else None
    try:
        with Path(path).open('rb') as source:
            while True:
                chunk = source.read(CHUNK)
                if not chunk:
                    break
                chash.update(chunk)
                pending = chunk
                while pending:
                    if decoder is None:
                        expanded, pending = pending, b''
                    else:
                        expanded = decoder.decompress(pending, min(CHUNK, size - count + 1))
                        pending = decoder.unconsumed_tail
                        if decoder.unused_data:
                            raise ValueError('Trailing DEFLATE bytes')
                    count += len(expanded)
                    if count > size:
                        raise ValueError('Expansion exceeds declared size')
                    output.write(expanded)
                    uhash.update(expanded)
                    crc = binascii.crc32(expanded, crc)
            if decoder is not None and (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
                raise ValueError('Truncated or trailing DEFLATE stream')
        if (count != size or crc & 0xffffffff != row['crc32']
                or chash.hexdigest() != row['compressedSHA256']
                or uhash.hexdigest() != row['uncompressedSHA256']):
            raise ValueError('Resource CRC/hash/size mismatch')
        output.seek(0)
        return output
    except BaseException:
        output.close()
        raise


def inference_whitelist(manifest):
    """Derive only known onboarding/test inputs; reject scored label fields."""
    result = set()
    allowed_test = {'frameId', 'logicalIndex', 'timestampSeconds', 'timestampKind',
                    'rgb', 'calibration', 'width', 'height', 'sessionId'}
    for obj in manifest['objects']:
        result.add((obj['model']['path'], 'model'))
        if len(obj['onboarding']) != 10 or len(obj['frames']) != 110:
            raise ValueError('Invalid onboarding/test split')
        for record in obj['onboarding']:
            result.add((record['rgb']['path'], 'rgb'))
            result.add((record['depth']['path'], 'onboarding_depth'))
        for record in obj['frames']:
            if set(record) - allowed_test:
                raise ValueError('Evaluator field in inference record')
            result.add((record['rgb']['path'], 'rgb'))
    return frozenset(result)


class ResourceStore:
    def __init__(self, root, storage, phase, whitelist=None, access_log=None):
        if phase not in PHASE_ROLES:
            raise ValueError('Unknown reader phase')
        if phase == 'inference' and whitelist is None:
            raise ValueError('Inference requires an explicit resource whitelist')
        self.root, self.phase = Path(root).resolve(), phase
        self.whitelist = frozenset(whitelist) if whitelist is not None else None
        self.access_log = access_log if access_log is not None else []
        self.index_path = inside(self.root, storage['indexPath'])
        self.index_sha = valid_sha(storage['indexSHA256'])
        with self.index_path.open('rb') as source:
            index_raw = source.read(8 * CHUNK + 1)
        if len(index_raw) > 8 * CHUNK or hashlib.sha256(index_raw).hexdigest() != self.index_sha:
            raise ValueError('Storage index seal changed')
        index = strict_json(index_raw.decode('utf-8'))
        if (index.get('schemaVersion') != 1 or index.get('datasetRevision') != storage['datasetRevision']
                or index.get('selectionDigest') != storage['selectionDigest']
                or index.get('corpus') != 'HB' or index.get('storageKind') != 'original_zip_members'):
            raise ValueError('Storage identity mismatch')
        self.entries = index['resources']
        identities = set()
        for logical, row in self.entries.items():
            safe_relative(logical)
            safe_relative(row['sourceMember'])
            identity = (row['datasetRevision'], row['archive'], row['sourceMember'])
            if identity in identities or row['datasetRevision'] != index['datasetRevision']:
                raise ValueError('Duplicate or conflicting resource identity')
            identities.add(identity)
            if (logical != row['sourceMember'] or type(row['crc32']) is not int
                    or not 0 <= row['crc32'] <= 0xffffffff):
                raise ValueError('Conflicting logical member or CRC metadata')
            if (row['sourceMethod'] not in (0, 8) or row['sourceFlags'] & ~2048
                    or type(row['sourceCompressedBytes']) is not int
                    or not 0 <= row['sourceCompressedBytes'] <= MAX_EXPANDED):
                raise ValueError('Unsupported original ZIP metadata')
            valid_sha(row['compressedSHA256'])
            valid_sha(row['uncompressedSHA256'])
            if not isinstance(row['allowedRoles'], list) or not row['allowedRoles']:
                raise ValueError('Resource roles missing')
            if row['location']['kind'] == 'payload':
                if row['method'] != row['sourceMethod'] or row['compressedBytes'] != row['sourceCompressedBytes']:
                    raise ValueError('Payload compression metadata changed')
                inside(self.root, row['location']['path'])
            elif row['location']['kind'] == 'reuse':
                if row['method'] != 0 or row['compressedSHA256'] != row['uncompressedSHA256']:
                    raise ValueError('Reused resources must bind exact expanded bytes')
                if not isinstance(row['location'].get('path'), str):
                    raise ValueError('Reuse path missing')
                # Reuse is explicit, hash-bound and read-only. Arbitrary external
                # specs cannot override this immutable index location.
                parts = PurePosixPath(logical).parts
                allowed = (logical == 'models/models_info.json' or
                    len(parts) == 3 and parts[0] == 'val_primesense' and
                    parts[2] in ('scene_camera.json', 'scene_gt.json', 'scene_gt_info.json'))
                if not allowed or Path(row['location']['path']).resolve(strict=True) != inside(REUSE_ROOT, logical):
                    raise ValueError('Reuse location is outside the explicit metadata allowlist')
            else:
                raise ValueError('Unknown storage location')

    def assert_sealed(self):
        if sha_file(self.index_path) != self.index_sha:
            raise ValueError('Storage index changed after sealing')

    def open_binary(self, resource, role):
        self.assert_sealed()
        logical = resource['path']
        safe_relative(logical)
        if role not in PHASE_ROLES[self.phase]:
            raise ValueError('Role forbidden in reader phase')
        if self.whitelist is not None and (logical, role) not in self.whitelist:
            raise ValueError('Resource absent from inference whitelist')
        row = self.entries[logical]
        if role not in row['allowedRoles'] or resource['sha256'] != row['uncompressedSHA256']:
            raise ValueError('Resource role or manifest hash mismatch')
        location = row['location']
        path = inside(self.root, location['path']) if location['kind'] == 'payload' else Path(location['path'])
        stream = verified_stream(path, row)
        self.access_log.append({'phase': self.phase, 'resource': logical, 'role': role,
                                'sha256': row['uncompressedSHA256']})
        return stream

    def open_text(self, resource, role, encoding='ascii'):
        return io.TextIOWrapper(self.open_binary(resource, role), encoding=encoding)


def provider(root, manifest, phase, access_log=None):
    storage = manifest.get('resourceStorage')
    if storage is None:
        return Path(root)
    receipt = strict_json((Path(root) / 'acquisition-receipt.json').read_text(encoding='utf-8'))
    expected_evaluator = (manifest['evaluatorManifestSHA256'] if phase == 'inference'
                          else sha_file(Path(root) / 'evaluator.json'))
    if (receipt.get('status') != 'complete' or receipt.get('datasetRevision') != storage['datasetRevision']
            or receipt.get('selectionDigest') != storage['selectionDigest']
            or receipt.get('indexSHA256') != storage['indexSHA256']
            or receipt.get('inferenceManifestSHA256') != sha_file(Path(root) / 'inference.json')
            or receipt.get('evaluatorManifestSHA256') != expected_evaluator):
        raise ValueError('Incomplete or changed compact acquisition receipt')
    return ResourceStore(root, storage, phase,
        inference_whitelist(manifest) if phase == 'inference' else None, access_log)
