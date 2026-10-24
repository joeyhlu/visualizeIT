"""Stage and verify the exact browser-only interactive segmentation runtime.

No package scripts are executed. The existing model is served in place; no
camera data or trained GoTrack weights are handled by this module.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

from tools.design_studio import _no_links, StudioError

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.1.1'
ARCHIVE_URL = f'https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{VERSION}.tgz'
ARCHIVE_INTEGRITY = 'jwntgT5tzDrKOkqg6xNDNpRcTUnGvmylZE5DFnzPXossiWOIDdZX+C81LI75hlXhsG58YZW+lGzcmjH1qrQE/Q=='
MODEL_SHA = '38431bc66b883404e8397f74c3579404315b9b52b04a46c6346fe906a7309b03'
MODEL_BYTES = 30525312
MANIFEST_SHA = '6720b9c275cd2dfd1f6416a6a12099d8ee43d4149fd2c1b333c5497e98644a31'
MODEL = ROOT / '.cache/show3d/models/interactive_segmentation.task'
RUNTIME = ROOT / '.cache/phone-camera-runtime' / VERSION
MEMBERS = ('vision_bundle.mjs', 'vision.d.ts', 'README.md',
           'wasm/vision_wasm_internal.js', 'wasm/vision_wasm_internal.wasm',
           'wasm/vision_wasm_module_internal.js', 'wasm/vision_wasm_module_internal.wasm',
           'wasm/vision_wasm_nosimd_internal.js', 'wasm/vision_wasm_nosimd_internal.wasm')
CAP = 64 << 20


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    _no_links(MODEL)
    model = MODEL.read_bytes()
    if len(model) != MODEL_BYTES or digest(model) != MODEL_SHA:
        raise ValueError('Cached v2 segmentation model does not match its pin')
    if shutil.disk_usage(ROOT).free < (1 << 30) + CAP:
        raise ValueError('Runtime staging requires 1 GiB reserve plus 64 MiB')
    with urllib.request.urlopen(ARCHIVE_URL, timeout=30) as response:
        archive = response.read(CAP + 1)
    if len(archive) > CAP or base64.b64encode(hashlib.sha512(archive).digest()).decode() != ARCHIVE_INTEGRITY:
        raise ValueError('Runtime archive integrity mismatch')
    assets = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as package:
        for name in MEMBERS:
            member = package.getmember('package/' + name)
            if not member.isfile() or member.size > CAP:
                raise ValueError('Unexpected runtime archive member')
            assets[name] = package.extractfile(member).read(CAP + 1)
    if sum(map(len, assets.values())) > CAP:
        raise ValueError('Runtime assets exceed staging allowance')
    for name, data in assets.items():
        target = RUNTIME / name
        _no_links(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f'Existing pinned runtime asset differs: {name}')
        else:
            with target.open('xb') as stream:
                stream.write(data)
    entries = {name: {'url': '/phone-runtime/' + name, 'sha256': digest(data), 'bytes': len(data)}
               for name, data in assets.items() if name not in ('README.md', 'vision.d.ts')}
    manifest = {'schema': 'visualizeit_phone_runtime_v1', 'runtime_version': VERSION,
                'api': 'interactive_segmenter_v2_strokes',
                'archive_url': ARCHIVE_URL, 'archive_sha512_base64': ARCHIVE_INTEGRITY,
                'model': {'url': '/phone-runtime/interactive_segmentation.task', 'sha256': MODEL_SHA, 'bytes': MODEL_BYTES},
                'module': entries['vision_bundle.mjs'], 'wasm_root': '/phone-runtime/wasm', 'assets': entries}
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    if digest(payload) != MANIFEST_SHA:
        raise ValueError('Generated runtime manifest differs from reviewed release pin')
    target = RUNTIME / 'manifest.json'
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError('Existing runtime manifest differs')
    else:
        with target.open('xb') as stream:
            stream.write(payload)
    return manifest


class PhoneRuntime:
    """Only exact manifest-listed public assets; no arbitrary file access."""
    def __init__(self):
        self.entries = {}
        self.manifest = None
        self.reason = 'Phone segmentation runtime has not been prepared.'
        try:
            path = _no_links(RUNTIME / 'manifest.json')
            encoded = path.read_bytes()
            if digest(encoded) != MANIFEST_SHA:
                raise ValueError('Runtime manifest hash mismatch')
            manifest = json.loads(encoded)
            if (manifest['schema'] != 'visualizeit_phone_runtime_v1' or manifest['runtime_version'] != VERSION
                    or manifest['model'] != {'url': '/phone-runtime/interactive_segmentation.task', 'sha256': MODEL_SHA, 'bytes': MODEL_BYTES}
                    or manifest['api'] != 'interactive_segmenter_v2_strokes'):
                raise ValueError('Runtime manifest identity mismatch')
            allowed = set(MEMBERS) - {'README.md', 'vision.d.ts'}
            if set(manifest['assets']) != allowed or manifest['module'] != manifest['assets']['vision_bundle.mjs']:
                raise ValueError('Runtime manifest asset mismatch')
            for name, item in manifest['assets'].items():
                if item['url'] != '/phone-runtime/' + name:
                    raise ValueError('Runtime asset URL mismatch')
                self.entries[item['url']] = (RUNTIME / name, item)
            self.entries[manifest['model']['url']] = (MODEL, manifest['model'])
            for asset_path, item in self.entries.values():
                self._read(asset_path, item)
            self.manifest = manifest
        except (OSError, ValueError, KeyError, TypeError, StudioError):
            self.entries = {}
            self.reason = 'Pinned phone segmentation runtime is missing or failed verification.'

    @staticmethod
    def _read(path, item):
        path = _no_links(path)
        if type(item['bytes']) is not int or not 0 < item['bytes'] <= CAP or path.stat().st_size != item['bytes']:
            raise ValueError('Runtime asset size mismatch')
        data = path.read_bytes()
        if digest(data) != item['sha256']:
            raise ValueError('Runtime asset hash mismatch')
        return data

    def get(self, route):
        if self.manifest is None:
            raise StudioError('phone_runtime_unavailable', self.reason, 503)
        if route == '/phone-runtime/manifest.json':
            return json.dumps(self.manifest).encode(), 'application/json'
        if route not in self.entries:
            raise StudioError('not_found', 'Phone runtime asset was not found.', 404)
        path, item = self.entries[route]
        try:
            data = self._read(path, item)
        except (ValueError, OSError, StudioError) as error:
            raise StudioError('phone_runtime_integrity', 'Phone runtime asset failed verification.', 503) from error
        mime = 'application/wasm' if path.suffix == '.wasm' else 'text/javascript' if path.suffix in ('.mjs', '.js') else 'application/octet-stream'
        return data, mime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.parse_args()
    result = prepare()
    print(json.dumps({'version': result['runtime_version'], 'model_sha256': MODEL_SHA,
                      'staged_bytes': sum(p.stat().st_size for p in RUNTIME.rglob('*') if p.is_file())}))
