"""Portable, input-only RAW capture producer. No import loads a neural model.

The prospective inventory is a recipe identity, never an execution inventory.
Generation sidecars remain outside the observed-input authorization boundary.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import shutil
import signal
import stat
import sys
import threading
import time

import numpy as np

from . import quality_adaptation_scale as scale
from .quality_kaggle_preflight import require_digest
from .quality_adaptation_data import sha256

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'portable-raw-v1'
SOURCE_LOCK_SHA256 = '44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33'
GiB = 1024**3
MAX_METADATA = 16 * 1024**2
MAX_ARTIFACT = 2 * GiB
CONTEXT_KEYS = {'schema', 'source_root', 'model_cache', 'output_root', 'scratch_root',
    'source_files', 'source_lock', 'device', 'limits', 'quota_scope', 'parity'}
LIMIT_KEYS = {'wall_seconds', 'rss_bytes', 'storage_bytes', 'reserve_bytes'}
_WITNESS_ISSUER = object()
_EXECUTION = ContextVar('portable_raw_stage', default=None)
_CAPTURE_SUCCESSOR = ContextVar('portable_raw_capture_successor', default=None)
EMERGENCY_METADATA_BYTES = 64 * 1024**2


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _loads(raw, name='portable metadata'):
    from .quality_gotrack_inference import _loads_unique
    if len(raw) > MAX_METADATA:
        raise ValueError(name + ' exceeds metadata bound')
    return _loads_unique(raw, name)


def _read_metadata_bytes(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_METADATA + 1)
    if len(raw) > MAX_METADATA:
        raise ValueError('portable metadata exceeds bounded read')
    return raw


def read_json(path):
    """Read portable metadata within the same 16 MiB contract as its writer.

    Retain the former reader's unique-key, finite-number and depth guards. Every
    portable metadata path uses this reader, including scale plans and raw seals.
    """
    value = _loads(_read_metadata_bytes(path), 'portable metadata')
    def finite_tree(item, depth=0):
        if depth > 32:
            raise ValueError('metadata JSON nesting exceeds 32')
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError('nonfinite JSON number')
        if isinstance(item, dict):
            for child in item.values():
                finite_tree(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                finite_tree(child, depth + 1)
    finite_tree(value)
    return value


def _keys(value, names, name):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(name + ' has unknown/missing fields')


def source_paths():
    from . import quality_adaptation_larger as larger
    return frozenset(larger.OPTIMIZED_SOURCE_FILES | {
        'bench/quality_adaptation_portable.py', 'bench/quality_adaptation_scale.py',
        'bench/quality_kaggle_preflight.py', 'scripts/kaggle_raw_prepare.py',
        'bench/quality_cnos.py', 'bench/quality_ranker_synthetic.py',
        'bench/quality_adaptation_runtime.py', 'bench/quality_adaptation_data.py',
        'bench/quality_adaptation_prepare.py', 'bench/quality_adaptation_expanded.py',
        'bench/quality_time.py', 'bench/quality_contract.py'})


def _absolute(value, name):
    if not isinstance(value, str) or not value or '..' in Path(value).parts or not Path(value).is_absolute():
        raise ValueError(name + ' requires an absolute path without traversal')
    path = Path(value)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(name + ' cannot traverse symlinks')
    return path.resolve()


def pin(path):
    path = Path(path)
    return dict(path=path.resolve().as_posix(), bytes=path.stat().st_size, sha256=sha256(path))


def check_pin(value, *, under=None, max_bytes=MAX_ARTIFACT):
    _keys(value, ('path', 'bytes', 'sha256'), 'artifact pin')
    path = _absolute(value['path'], 'artifact')
    require_digest(value['sha256'])
    if type(value['bytes']) is not int or not 0 < value['bytes'] <= max_bytes:
        raise ValueError('artifact pin size exceeds bound')
    if under is not None and not path.is_relative_to(Path(under).resolve()):
        raise ValueError('artifact escapes authorized root')
    if not path.is_file() or path.stat().st_size != value['bytes'] or sha256(path) != value['sha256']:
        raise ValueError('missing/altered pinned artifact: ' + path.name)
    return path


@dataclass(frozen=True, slots=True)
class PortableContext:
    """Detached immutable bytes; no caller-supplied verified flags or callbacks."""
    raw: bytes
    sha256: str

    def document(self):
        if type(self.raw) is not bytes or hashlib.sha256(self.raw).hexdigest() != self.sha256:
            raise ValueError('portable context bytes changed')
        return validate_context(_loads(self.raw, 'portable context'))


def read_context(path, expected_sha256):
    raw = _read_metadata_bytes(path)
    context = PortableContext(raw, require_digest(expected_sha256))
    context.document()
    return context


def validate_context(doc):
    _keys(doc, CONTEXT_KEYS, 'portable context')
    if doc['schema'] != 'kaggle_raw_context_v1' or doc['device'] not in ('cpu', 'cuda'):
        raise ValueError('unsupported portable context/device')
    root = _absolute(doc['source_root'], 'source root')
    cache = _absolute(doc['model_cache'], 'model cache')
    if root != ROOT.resolve() or cache != root / '.cache/model-quality':
        raise ValueError('source/cache must match actual existing loader locations')
    output = _absolute(doc['output_root'], 'output root')
    scratch = _absolute(doc['scratch_root'], 'scratch root')
    if (output == scratch or output.is_relative_to(scratch) or scratch.is_relative_to(output)
            or output.is_relative_to(cache) or scratch.is_relative_to(cache)
            or cache.is_relative_to(output) or cache.is_relative_to(scratch)):
        raise ValueError('portable writable/cache roots must be disjoint')
    if doc['quota_scope'] != 'one_shared_budget_across_output_scratch_and_model_cache_unknown_provider_quota_isolation':
        raise ValueError('explicit conservative quota scope required')
    limits = doc['limits']; _keys(limits, LIMIT_KEYS, 'portable limits')
    maxima = dict(wall_seconds=7200, rss_bytes=8 * GiB, storage_bytes=100 * GiB, reserve_bytes=8 * GiB)
    for name, maximum in maxima.items():
        if type(limits[name]) is not int or not 0 < limits[name] <= maximum:
            raise ValueError('invalid bounded ' + name)
    if set(doc['source_files']) != source_paths():
        raise ValueError('portable source closure differs')
    for relative, expected in doc['source_files'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('source traversal')
        require_digest(expected)
    _keys(doc['source_lock'], ('path', 'bytes', 'sha256'), 'immutable upstream lock')
    if (doc['source_lock']['path'] != (root / 'bench/runtime/adaptation-sources.json').as_posix()
            or doc['source_lock']['sha256'] != SOURCE_LOCK_SHA256):
        raise ValueError('immutable upstream source lock differs')
    # Parity evidence is bound now but opened only before actual neural work.
    _keys(doc['parity'], ('reference', 'candidate'), 'parity evidence')
    for value in doc['parity'].values():
        _keys(value, ('path', 'bytes', 'sha256'), 'parity pin')
        require_digest(value['sha256'])
    return doc


def context_document(*, output_root, scratch_root, parity, device='cuda',
        wall_seconds=7200, rss_bytes=8 * GiB, storage_bytes=18 * GiB, reserve_bytes=GiB):
    """Build reviewable metadata; does not install models or grant execution."""
    doc = dict(schema='kaggle_raw_context_v1', source_root=ROOT.resolve().as_posix(),
        model_cache=(ROOT / '.cache/model-quality').resolve().as_posix(),
        output_root=Path(output_root).resolve().as_posix(), scratch_root=Path(scratch_root).resolve().as_posix(),
        source_files={name: sha256(ROOT / name) for name in sorted(source_paths())},
        source_lock=pin(ROOT / 'bench/runtime/adaptation-sources.json'), device=device,
        limits=dict(wall_seconds=wall_seconds, rss_bytes=rss_bytes, storage_bytes=storage_bytes, reserve_bytes=reserve_bytes),
        quota_scope='one_shared_budget_across_output_scratch_and_model_cache_unknown_provider_quota_isolation',
        parity=parity)
    return validate_context(doc)


def verify_sources(context, *, upstream=False):
    if type(context) is not PortableContext:
        raise TypeError('exact immutable PortableContext required')
    doc = context.document()
    migration = _CAPTURE_SUCCESSOR.get()
    executing = doc
    if migration is not None and migration[0] == context.sha256:
        # Historical pins remain historical. Only the explicit continuation
        # scope may execute the separately authenticated successor sources.
        executing = migration[1].document()
    for relative, expected in executing['source_files'].items():
        path = _absolute((ROOT / relative).as_posix(), 'executing source')
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('portable executing source changed: ' + relative)
    lock = read_json(check_pin(doc['source_lock'], under=ROOT, max_bytes=MAX_METADATA))
    if upstream:
        from .quality_assets import SOURCES
        for kind in ('gotrack', 'sam2'):
            if lock[kind]['revision'] != SOURCES[kind]:
                raise ValueError('immutable upstream revision mismatch')
            for relative, expected in lock[kind]['files'].items():
                if relative.startswith('.cache/model-quality/sources/'):
                    path = _absolute((ROOT / relative).as_posix(), 'upstream source')
                    if not path.is_file() or sha256(path) != expected:
                        raise ValueError('immutable upstream bytes differ: ' + relative)
        allowed = {relative for kind in ('gotrack', 'sam2') for relative in lock[kind]['files']
            if relative.startswith('.cache/model-quality/sources/')}
        for path in (ROOT / '.cache/model-quality/sources').rglob('*'):
            if path.suffix in ('.py', '.yaml', '.yml') and path.relative_to(ROOT).as_posix() not in allowed:
                raise ValueError('unlisted executable upstream source: ' + path.name)
    return doc


def verify_neural_environment(context):
    """No acquisition helper; every prerequisite fails before model imports."""
    doc = verify_sources(context, upstream=True)
    # Do not let unverified cached bytecode override verified upstream sources.
    if any(Path(doc['model_cache']).joinpath('sources').rglob('*.pyc')):
        raise ValueError('upstream bytecode present; prepare a clean source-only runtime before execution')
    if any((ROOT / 'bench').rglob('*.pyc')) or any((ROOT / 'scripts').rglob('*.pyc')):
        raise ValueError('wrapper bytecode present; use a clean source package and python -B')
    from .quality_adaptation_runtime import _source_imports
    _source_imports('gotrack'); _source_imports('sam2')
    from .quality_assets import CACHE, MODELS
    if CACHE.resolve() != Path(doc['model_cache']):
        raise ValueError('actual model loader cache differs from authenticated cache')
    for name, entry in MODELS.items():
        path = CACHE / 'checkpoints' / name
        if path.is_symlink() or not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError('missing/altered pre-existing model; no downloads: ' + name)
    lock = read_json(CACHE / 'lock.json')
    for name, entry in MODELS.items():
        if lock.get('models', {}).get(name, {}).get('sha256') != entry['sha256']:
            raise ValueError('actual descriptor loader lock mismatch')
    from .quality_adaptation_runtime import runtime_versions
    versions = runtime_versions(sam=True)
    from .quality_adaptation_larger_parity import compare_runs
    parity = compare_runs(*[read_json(check_pin(doc['parity'][key], max_bytes=MAX_METADATA))
        for key in ('reference', 'candidate')])
    if read_json(check_pin(doc['parity']['candidate'], max_bytes=MAX_METADATA))['device'] != doc['device']:
        raise ValueError('parity device differs from selected execution')
    return dict(versions=versions, parity=parity)


@contextmanager
def _storage_windows_directory(path):
    """Hold a directory against replacement while metadata-only Windows scans run.

    Excluding FILE_SHARE_DELETE prevents a directory from becoming a junction or
    symlink between inspection and scandir. Parents remain held by the caller.
    Cloud execution uses the descriptor-relative POSIX path below.
    """
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,); close.restype = wintypes.BOOL
    info = kernel.GetFileInformationByHandleEx
    info.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    info.restype = wintypes.BOOL
    class Attributes(ctypes.Structure):
        _fields_ = [('attributes', wintypes.DWORD), ('reparse_tag', wintypes.DWORD)]
    # LIST_DIRECTORY|READ_ATTRIBUTES, SHARE_READ|SHARE_WRITE, OPEN_EXISTING,
    # BACKUP_SEMANTICS|OPEN_REPARSE_POINT. Never open a reparse target.
    # Metadata-only access does not participate in Windows sharing checks.
    handle = create(str(path), 0x81, 3, None, 3, 0x02200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        value = Attributes()
        if not info(handle, 9, ctypes.byref(value), ctypes.sizeof(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        if value.attributes & 0x400:
            raise ValueError('symlink/reparse point in bounded storage tree')
        if not value.attributes & 0x10:
            raise ValueError('non-directory in bounded storage traversal')
        yield
    finally:
        close(handle)


def _storage_scan(roots):
    """One complete no-follow traversal; any vanished descendant invalidates it."""
    from contextlib import ExitStack
    seen, directories = set(), set()
    total = 0
    def account(value):
        nonlocal total
        identity = value.st_dev, value.st_ino
        if stat.S_ISLNK(value.st_mode):
            raise ValueError('symlink in bounded storage tree')
        if stat.S_ISREG(value.st_mode):
            if identity not in seen:
                total += value.st_size; seen.add(identity)
        elif not stat.S_ISDIR(value.st_mode):
            raise ValueError('nonregular path in bounded storage tree')
        return identity

    if os.name == 'nt':
        def visit(path):
            with _storage_windows_directory(path):
                identity = account(path.lstat())
                if identity in directories:
                    return
                directories.add(identity)
                with os.scandir(path) as entries:
                    for entry in entries:
                        child = path / entry.name
                        value = child.lstat(); account(value)
                        if stat.S_ISDIR(value.st_mode):
                            visit(child)
        for root in roots:
            root = Path(root).absolute()
            try:
                value = root.lstat()
            except FileNotFoundError:
                continue  # A prospective root may not have been created yet.
            account(value)
            if stat.S_ISDIR(value.st_mode):
                with ExitStack() as stack:
                    for ancestor in reversed(root.parents):
                        stack.enter_context(_storage_windows_directory(ancestor))
                    visit(root)
    else:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        @contextmanager
        def opened(name, parent=None):
            try:
                descriptor = os.open(name, flags, dir_fd=parent)
            except NotADirectoryError as error:
                raise ValueError('symlink/non-directory in bounded storage tree') from error
            try:
                yield descriptor
            finally:
                os.close(descriptor)
        def visit(descriptor):
            identity = account(os.fstat(descriptor))
            if identity in directories:
                return
            directories.add(identity)
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    value = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                    account(value)
                    if stat.S_ISDIR(value.st_mode):
                        with opened(entry.name, descriptor) as child:
                            if account(os.fstat(child)) != (value.st_dev, value.st_ino):
                                raise FileNotFoundError('directory changed during storage scan')
                            visit(child)
                        after = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                        if account(after) != (value.st_dev, value.st_ino):
                            raise FileNotFoundError('directory changed during storage scan')
        for root in roots:
            root = Path(root).absolute()
            try:
                value = root.lstat()
            except FileNotFoundError:
                continue
            account(value)
            if not stat.S_ISDIR(value.st_mode):
                continue
            with ExitStack() as stack:
                descriptor = stack.enter_context(opened(root.anchor))
                for component in root.parts[1:]:
                    descriptor = stack.enter_context(opened(component, descriptor))
                visit(descriptor)
                after = root.lstat()
                if account(after) != (value.st_dev, value.st_ino):
                    raise FileNotFoundError('quota root changed during storage scan')
    return total


def _usage(roots):
    """Count each regular inode once; retry entire snapshots after atomic renames.

    A partial sum is never accepted. Three unstable traversals cancel the stage;
    symlinks, permission errors and other filesystem failures are never ignored.
    """
    roots = tuple(roots)
    for attempt in range(3):
        try:
            return _storage_scan(roots)
        except FileNotFoundError as error:
            if attempt == 2:
                raise ResourceError('bounded storage tree remained unstable after three scans') from error


def resource_check(context, reserve=0, *, metadata_write=False):
    doc = context.document(); limits = doc['limits']
    roots = [Path(doc[key]) for key in ('output_root', 'scratch_root', 'model_cache')]
    used = _usage(roots)
    if used + reserve + (0 if metadata_write else EMERGENCY_METADATA_BYTES) > limits['storage_bytes']:
        raise ResourceError('shared portable storage bound exhausted')
    if _usage([roots[-1]]) > 8 * GiB:
        raise ResourceError('8 GiB source/checkpoint cache bound exhausted')
    checked = set()
    for path in roots:
        while not path.exists():
            path = path.parent
        device = path.stat().st_dev
        if device not in checked and shutil.disk_usage(path).free < reserve + limits['reserve_bytes']:
            raise ResourceError('physical filesystem reserve exhausted')
        checked.add(device)
    return dict(shared_physical_bytes=used, devices=sorted(checked), quota_isolation='unknown')


class ResourceError(BaseException):
    """Cancellation, like KeyboardInterrupt: row failure handlers cannot swallow it."""


def write_json(context, path, value, *, replace=False):
    """Atomic bounded metadata writes; a prior attempt is never overwritten by default."""
    return write_bytes(context, path, encoded(value), replace=replace)


def write_bytes(context, path, raw, *, replace=False):
    """Preserve exact authenticated context/recipe bytes, including whitespace."""
    doc = context.document(); root = Path(doc['output_root']); path = Path(path)
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('metadata output escapes run')
    if len(raw) > MAX_METADATA:
        raise ResourceError('metadata bound exhausted')
    resource_check(context, len(raw), metadata_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError('attempt already recorded; preserve original')
    temporary = path.with_name('.' + path.name + '.part')
    temporary_created = False
    try:
        with temporary.open('xb') as stream:
            temporary_created = True
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if path.exists() and not replace:
            raise FileExistsError('attempt appeared during write')
        temporary.replace(path)
    finally:
        if temporary_created:
            temporary.unlink(missing_ok=True)
    return pin(path)


def _failure(exc):
    return f'{type(exc).__name__}: {exc}'[:1024]


def prospective(spec):
    """Freeze every permitted asset/view/state before any mesh payload is opened."""
    _keys(spec, ('schema', 'assets', 'iterations', 'seed', 'targets'), 'portable recipe')
    if spec['schema'] != 'kaggle_raw_recipe_v1':
        raise ValueError('unsupported raw recipe')
    iterations = spec['iterations']
    if (not isinstance(iterations, list) or not iterations or iterations != sorted(set(iterations))
            or any(type(i) is not int or i not in (0, 1, 3) for i in iterations)):
        raise ValueError('only existing selected iterations 0/1/3 allowed')
    if not isinstance(spec['assets'], list) or not spec['assets']:
        raise ValueError('complete prospective asset inventory required')
    ids, meshes, records = set(), set(), []
    for asset in spec['assets']:
        _keys(asset, ('object_id', 'physical_object_id', 'aliases', 'mesh'), 'prospective asset')
        obj = asset['object_id']
        if (not isinstance(obj, str) or not obj.isdecimal() or str(int(obj)) != obj or
                obj in ids or not isinstance(asset['physical_object_id'], str) or not asset['physical_object_id']):
            raise ValueError('unique normalized numeric object IDs required')
        ids.add(obj)
        _keys(asset['mesh'], ('path', 'bytes', 'sha256'), 'prospective mesh pin')
        require_digest(asset['mesh']['sha256'])
        mesh_path = asset['mesh']['path']
        if (not isinstance(mesh_path, str) or not (Path(mesh_path).is_absolute() or PurePosixPath(mesh_path).is_absolute())
                or Path(mesh_path).suffix.lower() != '.glb' or '..' in Path(mesh_path).parts
                or type(asset['mesh']['bytes']) is not int or not 0 < asset['mesh']['bytes'] <= 64 * 1024**2):
            raise ValueError('prospective metric GLB path/size exceeds existing renderer contract')
        if asset['mesh']['sha256'] in meshes:
            raise ValueError('repeated metric mesh cannot enlarge the inventory')
        meshes.add(asset['mesh']['sha256'])
        if not isinstance(asset['aliases'], list) or any(not isinstance(x, str) for x in asset['aliases']):
            raise ValueError('asset aliases required')
        for view in range(24):
            sequence = f'portable-raw-obj{obj}-view{view:02d}'
            recipe = dict(mesh_sha256=asset['mesh']['sha256'], view_mode='normalized24-v1',
                view_index=view, native_resolution=[640, 480], fps=10., frame_id=0, timestamp_s=0.)
            for iteration in iterations:
                records.append(dict(id=sequence + f':frame0:iteration{iteration}',
                    physical_object_id=asset['physical_object_id'], aliases=asset['aliases'],
                    sequence_id=sequence, scene_id='portable-procedural-object:' + obj,
                    background_id='procedural:gray', background_kind='procedural', provenance_kind='synthetic',
                    frame_id=0, source_frame_id=0, mesh_sha256=asset['mesh']['sha256'],
                    refinement_state=iteration, inputs_sha256=scale.canonical_hash(dict(recipe, iteration=iteration))))
    return scale.plan_scale(dict(records=records,
        label_policy=dict(max_rotation_error_deg=10., max_translation_error_m=.02)),
        seed=spec['seed'], targets=spec['targets'])


def physical_rows(spec, plan):
    states = plan['global_manifest']['records']; rows = []
    for asset in spec['assets']:
        for view in range(24):
            sequence = f"portable-raw-obj{asset['object_id']}-view{view:02d}"
            selected = [s for s in states if s['sequence_id'] == sequence]
            rows.append(dict(id=sequence + ':frame0', object_id=asset['object_id'],
                physical_object_id=asset['physical_object_id'], sequence_id=sequence,
                capture_group=selected[0]['scene_id'], split=selected[0]['split'],
                frame_id=0, timestamp_s=0., view_index=view, iterations=spec['iterations'],
                mesh_recipe=asset['mesh'], render=dict(status='pending'),
                bank=dict(status='pending'), mask=dict(status='pending'), seed=dict(status='pending')))
    return rows


def freeze(spec_path, expected_sha256, context):
    began = time.monotonic()
    doc = verify_sources(context)
    raw = _read_metadata_bytes(spec_path)
    if hashlib.sha256(raw).hexdigest() != require_digest(expected_sha256):
        raise ValueError('prospective recipe bytes changed')
    spec = _loads(raw, 'portable recipe'); plan = prospective(spec)
    root = Path(doc['output_root'])
    if root.exists():
        raise FileExistsError('fresh portable run required')
    root.mkdir(parents=True)
    write_bytes(context, root / 'context.json', context.raw)
    write_bytes(context, root / 'recipe.json', raw)
    write_json(context, root / 'scale-plan.json', plan)
    write_json(context, root / 'split.json', plan['global_manifest']['split_manifest'])
    write_json(context, root / 'physical-index.json', dict(rows=physical_rows(spec, plan)))
    receipt = dict(schema='kaggle_raw_freeze_v1', context_sha256=context.sha256,
        recipe_sha256=expected_sha256, recipe_file_sha256=sha256(root / 'recipe.json'),
        scale_plan_sha256=sha256(root / 'scale-plan.json'),
        physical_frames=len(spec['assets']) * 24, refinement_states=len(plan['global_manifest']['records']),
        prospective_inputs_are_recipe_identity_only=True, training_ready=False,
        elapsed_seconds=time.monotonic() - began, created_unix_s=time.time())
    write_json(context, root / 'freeze.json', receipt)
    return receipt


def frozen(context):
    doc = verify_sources(context); root = Path(doc['output_root'])
    receipt = read_json(root / 'freeze.json')
    if receipt['context_sha256'] != context.sha256 or sha256(root / 'context.json') != context.sha256:
        raise ValueError('frozen context differs')
    if sha256(root / 'recipe.json') != receipt['recipe_file_sha256']:
        raise ValueError('frozen recipe differs')
    plan = scale.validate_plan(read_json(root / 'scale-plan.json'))
    if (sha256(root / 'scale-plan.json') != receipt['scale_plan_sha256'] or
            read_json(root / 'split.json') != plan['global_manifest']['split_manifest']):
        raise ValueError('prospective split/inventory differs')
    spec = read_json(root / 'recipe.json')
    if prospective(spec) != plan:
        raise ValueError('prospective recipe/inventory disagree')
    rows = read_json(root / 'physical-index.json')['rows']
    original = physical_rows(spec, plan)
    if len(rows) != len(original):
        raise ValueError('physical frame inventory changed')
    for row, expected in zip(rows, original):
        if set(row) != set(expected) or any(row[k] != v for k, v in expected.items()
                if k not in ('render', 'bank', 'mask', 'seed')):
            raise ValueError('physical frame identity changed')
        for key in ('render', 'bank', 'mask', 'seed'):
            state = row[key]
            if not isinstance(state, dict) or state.get('status') not in ('pending', 'available', 'unavailable'):
                raise ValueError('invalid physical availability state')
            if state['status'] == 'unavailable' and not state.get('failure_reason'):
                raise ValueError('physical failure requires observable reason')
        if row['split'] in ('test', 'dev') and any(row[k] != {'status': 'pending'}
                for k in ('render', 'bank', 'mask', 'seed')):
            raise ValueError('sealed TEST/dev payload attempt forbidden')
    return doc, spec, plan, rows


def _active(rows):
    return [row for row in rows if row['split'] in ('train', 'validation')]


def _view_directory(root, row):
    return Path(root) / 'views' / f"obj-{int(row['object_id']):06d}-view-{row['view_index']:02d}"


def _save_rows(context, rows):
    write_json(context, Path(context.document()['output_root']) / 'physical-index.json', dict(rows=rows), replace=True)


def _rss_tree():
    """Current Linux process plus descendants; no optional psutil dependency."""
    processes = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
            processes[int(path.parent.name)] = (int(fields['PPid']), int(fields.get('VmRSS', '0 kB').split()[0]) * 1024)
        except (OSError, ValueError, KeyError):
            continue
    descendants, pending = set(), [os.getpid()]
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid); pending.extend(key for key, (ppid, _) in processes.items() if ppid == pid)
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)


def _require_stage(context, name):
    if _EXECUTION.get() != (context.sha256, name):
        raise RuntimeError('call this producer inside its explicit guarded stage')


def _close_neighbors():
    from . import quality_neighbors
    worker = quality_neighbors._client
    if worker is not None:
        worker.close()
        quality_neighbors._client = None


def _record_stage_failure(context, name, error):
    doc, spec, plan, rows = frozen(context); root = Path(doc['output_root'])
    key = {'render': 'render', 'banks': 'bank', 'masks': 'mask', 'capture': 'seed'}.get(name)
    if key is not None:
        for row in _active(rows):
            if row[key]['status'] == 'pending':
                row[key] = dict(status='unavailable', failure_reason=error)
        _save_rows(context, rows)
    if name == 'capture' and not (root / 'raw/capture-receipt.json').exists():
        from .quality_adaptation_refinement_states import state_rows
        states = state_rows(dict(rows=rows))
        for state in states:
            state['failure_reason'] = error if state['split'] in ('train', 'validation') else 'sealed_metadata_only'
        if not (root / 'input-plan.json').exists():
            write_json(context, root / 'input-plan.json', capture_plan(context, rows))
        write_json(context, root / 'raw/capture-index.json', dict(rows=states, training_ready=False), replace=True)
        write_json(context, root / 'raw/capture-receipt.json', dict(schema='kaggle_raw_capture_receipt_v1',
            input_plan_sha256=sha256(root / 'input-plan.json'), context_sha256=context.sha256,
            fatal_error=error, cleanup_errors=[], physical_frames_requested=len(rows),
            refinement_states_requested=len(states), annotation_inputs=[], test_payload_reads=0, training_ready=False))


@contextmanager
def stage(context, name, *, cloud_execution=False):
    from . import quality_adaptation_larger as larger
    began = time.monotonic()
    doc = context.document()
    if (not cloud_execution or platform.system() != 'Linux' or
            os.environ.get('VISUALIZEIT_CLOUD') != '1' or not Path('/kaggle/input').is_dir() or
            not Path(doc['output_root']).is_relative_to('/kaggle/working') or
            not Path(doc['scratch_root']).is_relative_to('/tmp')):
        raise RuntimeError('explicit Kaggle Linux cloud execution and scoped working/tmp roots required')
    root = Path(doc['output_root'])
    names = ('render', 'banks', 'masks', 'capture', 'seal-capture')
    if name not in names:
        raise ValueError('unknown portable stage')
    index = names.index(name)
    if index and not (root / ('stage-' + names[index - 1] + '.json')).is_file():
        raise ValueError('previous stage must be attempted before this stage')
    if (root / 'raw-seal.json').exists():
        raise ValueError('raw run already sealed')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('stage watchdog requires main thread')
    elapsed = read_json(root / 'freeze.json')['elapsed_seconds']
    for path in root.glob('stage-*.json'):
        previous = read_json(path)
        if previous.get('context_sha256') != context.sha256:
            raise ValueError('previous stage context differs')
        elapsed += previous['elapsed_seconds']
    remaining = doc['limits']['wall_seconds'] - elapsed - (time.monotonic() - began)
    if remaining <= 0:
        raise ResourceError('cumulative execution deadline exhausted')
    stage_path = root / ('stage-' + name + '.json')
    if stage_path.exists():
        raise FileExistsError('stage already attempted; preserve failures')
    peak, status, error = 0, 'failed', None
    stopped = threading.Event(); problems = []
    def interrupt(signum, frame):
        raise ResourceError(problems[0] if problems else 'cumulative execution deadline exhausted')
    def monitor():
        nonlocal peak
        while not stopped.wait(.1):
            try:
                peak = max(peak, _rss_tree())
                if peak > doc['limits']['rss_bytes']:
                    raise ResourceError('process tree RSS bound exhausted')
                resource_check(context)
            except BaseException as exc:
                if not stopped.is_set():
                    problems.append(_failure(exc)); os.kill(os.getpid(), signal.SIGALRM)
                return
    with larger.stage_lock():
        old_handler = signal.getsignal(signal.SIGALRM)
        if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
            raise RuntimeError('another execution deadline is active')
        signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        token = _EXECUTION.set((context.sha256, name))
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        try:
            resource_check(context)
            frozen(context)
            if name in ('render', 'banks', 'masks', 'capture'):
                verify_neural_environment(context)
            yield
            resource_check(context)
            status = 'completed'
        except BaseException as exc:
            error = _failure(exc)
            _record_stage_failure(context, name, error)
            raise
        finally:
            stopped.set(); watcher.join(timeout=2)
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, old_handler)
            _EXECUTION.reset(token)
            cleanup_error = None
            try:
                _close_neighbors()
            except BaseException as exc:
                status, error, cleanup_error = 'failed', _failure(exc), exc
            write_json(context, stage_path, dict(stage=name, context_sha256=context.sha256,
                status=status, error=error,
                elapsed_seconds=time.monotonic() - began, process_tree_peak_rss_bytes=peak,
                quota_scope=doc['quota_scope'], test_payload_reads=0, training_ready=False))
            if cleanup_error is not None:
                raise RuntimeError('stage worker cleanup failed: ' + error) from cleanup_error


def render(context):
    _require_stage(context, 'render')
    from .quality_ranker_synthetic import _render_bundle
    doc, spec, plan, rows = frozen(context); root = Path(doc['output_root'])
    groups = {g['sequence_id']: g for g in plan['global_manifest']['split_manifest']['groups']}
    try:
        for row in _active(rows):
            try:
                resource_check(context, 576 * 1024**2)
                result = _render_bundle(check_pin(row['mesh_recipe']), _view_directory(root, row),
                    groups[row['sequence_id']], frames=1, width=640, height=480, fps=10.,
                    mesh_units='metres', observationally_ambiguous=True, cloud_execution=True,
                    split_manifest_path=root / 'split.json', view_mode='normalized24-v1', view_index=row['view_index'])
                bundle = _view_directory(root, row) / 'bundle'
                row['render'] = dict(status='available', input=pin(bundle / 'input.json'),
                    video=pin(bundle / 'synthetic.mp4'), mesh=pin(bundle / 'object.glb'))
                # Generation receipt is deliberately not part of inference context.
                write_json(context, root / f"render-{int(row['object_id']):06d}-{row['view_index']:02d}.json", row['render'])
            except ResourceError:
                raise
            except Exception as exc:
                row['render'] = dict(status='unavailable', failure_reason=_failure(exc))
            _save_rows(context, rows)
    finally:
        for row in _active(rows):
            if row['render']['status'] == 'pending':
                row['render'] = dict(status='unavailable', failure_reason='unrun_after_termination')
        _save_rows(context, rows)
    return dict(physical_frames_requested=len(rows), rendered=sum(r['render']['status'] == 'available' for r in rows))


def read_observed_input(context, row, *, decode=True):
    from .quality_runner import read_input, read_rgb
    doc = context.document()
    if row['split'] not in ('train', 'validation') or row['render']['status'] != 'available':
        raise ValueError('TEST/dev/unavailable observed payload denied')
    receipt = row['render']
    for key in ('input', 'video', 'mesh'):
        check_pin(receipt[key], under=doc['output_root'])
    bundle = Path(receipt['input']['path']).parent
    if (Path(receipt['mesh']['path']) != bundle / 'object.glb' or
            Path(receipt['video']['path']) != bundle / 'synthetic.mp4'):
        raise ValueError('observed resource must be inside its own bundle')
    manifest = read_input(bundle)
    from . import quality_gotrack_inference as inference
    allowed = {'schema_version', 'object', 'object_id', 'physical_object_id', 'scene_id',
        'sequence_id', 'background_id', 'background_kind', 'provenance_kind', 'units',
        'video', 'asset', 'source_hashes', 'intrinsics', 'native_resolution',
        'setup_frame_id', 'frame_ids', 'clock', 'timeline', '_timing'}
    if set(manifest) != allowed:
        raise ValueError('foreign/annotation fields forbidden in inference input')
    if (manifest.get('schema_version') != 2 or manifest.get('provenance_kind') != 'synthetic'
            or manifest.get('units') != 'metres' or manifest.get('asset') != 'object.glb'
            or manifest.get('video') != 'synthetic.mp4' or manifest.get('_capture') is not None
            or manifest.get('physical_object_id') != row['physical_object_id']
            or manifest.get('scene_id') != row['capture_group']
            or manifest.get('background_id') != 'procedural:gray' or manifest.get('background_kind') != 'procedural'
            or manifest.get('sequence_id') != row['sequence_id'] or manifest.get('frame_ids') != [0]
            or manifest.get('native_resolution') != [640, 480]
            or manifest['_timing']['clock_mode'] != 'capture_seconds_v1'
            or manifest.get('source_hashes') != {'object.glb': receipt['mesh']['sha256'],
                'synthetic.mp4': receipt['video']['sha256']}
            or receipt['mesh']['sha256'] != row['mesh_recipe']['sha256']):
        raise ValueError('portable observed input differs from frozen recipe')
    keys = manifest['_timing']['scored_keys']
    if len(keys) != 1 or keys[0].timestamp_s != 0. or keys[0].frame_id != 0:
        raise ValueError('physical timestamp differs from frozen normalized view')
    inference._validated_native_intrinsics(manifest, {})
    return receipt, bundle, manifest, read_rgb(bundle, manifest, 0) if decode else None


def banks(context):
    _require_stage(context, 'banks')
    from . import quality_adaptation_larger as larger
    from .quality_gotrack import load_network, TexturedRenderer
    from .quality_foundpose import build_bank
    from .quality_cnos import build_descriptors
    doc, spec, plan, rows = frozen(context); root = Path(doc['output_root']); device = doc['device']
    torch = larger.configure_torch(device, deterministic=True)
    try:
        for obj in dict.fromkeys(row['object_id'] for row in _active(rows)):
            selected = [r for r in _active(rows) if r['object_id'] == obj]
            usable = [r for r in selected if r['render']['status'] == 'available']
            network = renderer = state = None
            try:
                if not usable:
                    raise ValueError('no rendered input available for object bank')
                resource_check(context, MAX_ARTIFACT)
                _, bundle, manifest, rgb = read_observed_input(context, usable[0]); rgb = None
                cnos_path = root / 'banks' / f'cnos-{int(obj):06d}.npz'
                pose_path = root / 'banks' / f'pose-{int(obj):06d}.pt'
                if cnos_path.exists() or pose_path.exists() or pose_path.with_suffix('.part').exists():
                    raise FileExistsError('bank output exists')
                # Descriptor model has left scope and CUDA is cleared before pose residency.
                build_descriptors(bundle, device, unlit=True, output=cnos_path)
                larger.cleanup_device(device, torch)
                write_json(context, root / f'cnos-{int(obj):06d}.json', dict(bank=pin(cnos_path)))
                network = load_network(device).eval()
                renderer = TexturedRenderer(bundle / 'object.glb', manifest['object_id'], unlit=True)
                state = build_bank(renderer, network.backbone, device, pose_path)
                if state.get('template_count') != 798 or state.get('asset_sha256') != usable[0]['mesh_recipe']['sha256']:
                    raise ValueError('standard immutable FoundPose bank differs')
                result = dict(status='available', bank=pin(pose_path), cnos_bank=pin(cnos_path))
            except ResourceError:
                raise
            except Exception as exc:
                result = dict(status='unavailable', failure_reason=_failure(exc))
            finally:
                state = None
                cleanup_errors = []
                for action in (lambda: renderer.close() if renderer is not None else None,
                        lambda: network.cpu() if network is not None else None,
                        lambda: larger.cleanup_device(device, torch)):
                    try:
                        action()
                    except BaseException as exc:
                        cleanup_errors.append(_failure(exc))
                renderer = network = None
                if cleanup_errors:
                    raise RuntimeError('bank cleanup failed: ' + '; '.join(cleanup_errors))
            for row in selected:
                row['bank'] = result.copy()
            _save_rows(context, rows)
    finally:
        for row in _active(rows):
            if row['bank']['status'] == 'pending':
                row['bank'] = dict(status='unavailable', failure_reason='unrun_after_termination')
        _save_rows(context, rows)


def _write_mask(context, physical, selected, diagnostics):
    from .vision import cv2
    doc = context.document(); root = Path(doc['output_root'])
    resource_check(context, 2 * 1024**2)
    directory = _view_directory(root, physical) / 'observed-masks'
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / '0.png'
    if not cv2.imwrite(str(path), selected.astype(np.uint8) * 255):
        raise OSError('observed SAM mask write failed')
    receipt = dict(schema='kaggle_raw_observed_mask_v1', automatic=True,
        mask_origin='observed_CNOS_box_then_SAM2_image_predictor',
        context_sha256=context.sha256, physical_id=physical['id'],
        input=physical['render']['input'], video=physical['render']['video'],
        mesh=physical['render']['mesh'], cnos_bank=physical['bank']['cnos_bank'], mask=pin(path),
        detector=diagnostics['detector'], prompt_box_xyxy=diagnostics['prompt_box_xyxy'],
        proposal_intermediate_sha256=diagnostics['proposals_sha256'], annotation_inputs=[], test_payload_reads=0)
    index = write_json(context, directory / 'results.json', receipt)
    return dict(mask=pin(path), mask_index=index)


def masks(context):
    _require_stage(context, 'masks')
    from . import quality_adaptation_larger as larger
    from .quality_adaptation_larger_optimized import NeuralMaskBackend, run_mask_phases
    doc, spec, plan, rows = frozen(context); root = Path(doc['output_root'])
    torch = larger.configure_torch(doc['device'], deterministic=True)
    try:
        for obj in dict.fromkeys(r['object_id'] for r in _active(rows)):
            selected = [r for r in _active(rows) if r['object_id'] == obj]
            views = [r for r in selected if r['render']['status'] == r['bank']['status'] == 'available']
            for row in selected:
                if row not in views:
                    row['mask'] = dict(status='unavailable', failure_reason='render_or_bank_unavailable')
            _save_rows(context, rows)
            if not views:
                continue
            by_sequence = {v['sequence_id']: v for v in views}
            backend_doc = dict(assets=spec['assets'])
            backend = NeuralMaskBackend(root, backend_doc, obj, doc['device'], torch,
                Path(doc['model_cache']), pin_resolver=lambda value: check_pin(value, under=root),
                input_reader=lambda view: read_observed_input(context, by_sequence[view['sequence_id']]),
                mask_writer=lambda view, selected_mask, diagnostic: _write_mask(context,
                    by_sequence[view['sequence_id']], selected_mask, diagnostic))
            def persist(results, summary):
                for result in results:
                    row = by_sequence[result['view']['sequence_id']]
                    row['mask'] = dict(status=result['status'], failure_reason=result['failure_reason'], **result.get('output', {}))
                _save_rows(context, rows)
                write_json(context, root / f'mask-phases-{int(obj):06d}.json', summary)
            run_mask_phases(views, Path(doc['scratch_root']) / ('proposals-' + obj), backend,
                reserve=lambda amount: resource_check(context, amount), persist=persist)
    finally:
        try:
            larger.cleanup_device(doc['device'], torch)
        finally:
            for row in _active(rows):
                if row['mask']['status'] == 'pending':
                    row['mask'] = dict(status='unavailable', failure_reason='unrun_after_termination')
            _save_rows(context, rows)


def capture_plan(context, rows):
    """Strict input-only projection. No generation sidecar paths survive."""
    return dict(schema='kaggle_raw_input_plan_v1', context_sha256=context.sha256,
        rows=[{key: value for key, value in row.items() if key != 'seed'} for row in rows])


def _validate_capture_plan(plan, context):
    _keys(plan, ('schema', 'context_sha256', 'rows'), 'portable capture plan')
    if plan['schema'] != 'kaggle_raw_input_plan_v1' or plan['context_sha256'] != context.sha256:
        raise ValueError('portable capture context mismatch')
    doc, spec, frozen_plan, rows = frozen(context)
    expected = capture_plan(context, rows)
    if plan != expected:
        raise ValueError('capture plan differs from frozen physical input inventory')
    return rows


def preflight_inputs(raw, expected_sha256, context):
    """Validate the entire metadata inventory before any TRAIN/VAL payload opens."""
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != require_digest(expected_sha256):
        raise ValueError('portable input plan bytes changed')
    verify_sources(context, upstream=True)
    plan = _loads(raw, 'portable capture plan'); rows = _validate_capture_plan(plan, context)
    doc = context.document(); root = Path(doc['output_root']); inputs = {}
    # Invocation-local deduplication only: every new preflight hashes anew.
    verified_pins = {}
    def verify_once(value):
        key = (value['path'], value['bytes'], value['sha256'])
        if key not in verified_pins:
            verified_pins[key] = check_pin(value, under=root)
        return verified_pins[key]
    # First pass rejects sealed payload pins and metadata changes, including LAST row.
    for row in rows:
        if row['split'] not in ('train', 'validation'):
            if any(row[key] != {'status': 'pending'} for key in ('render', 'bank', 'mask', 'seed')):
                raise ValueError('TEST/dev payload opening denied')
    for row in _active(rows):
        if any(row[key]['status'] != 'available' for key in ('render', 'bank', 'mask')):
            continue
        _, bundle, manifest, _ = read_observed_input(context, row, decode=False)
        for key in ('bank', 'cnos_bank'):
            verify_once(row['bank'][key])
        index = read_json(check_pin(row['mask']['mask_index'], under=root, max_bytes=MAX_METADATA))
        expected = dict(schema='kaggle_raw_observed_mask_v1', automatic=True,
            mask_origin='observed_CNOS_box_then_SAM2_image_predictor', context_sha256=context.sha256,
            physical_id=row['id'], **{k: row['render'][k] for k in ('input', 'video', 'mesh')},
            cnos_bank=row['bank']['cnos_bank'], mask=row['mask']['mask'], annotation_inputs=[], test_payload_reads=0)
        if set(index) != set(expected) | {'detector', 'prompt_box_xyxy', 'proposal_intermediate_sha256'} or any(
                index.get(k) != value for k, value in expected.items()):
            raise ValueError('observed mask producer/context binding differs')
        require_digest(index['proposal_intermediate_sha256'])
        box = index['prompt_box_xyxy']
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0] <= box[2] < 640 or not 0 <= box[1] <= box[3] < 480
                or index['detector'].get('bank_sha256') != row['bank']['cnos_bank']['sha256']):
            raise ValueError('observed association binding differs')
        from .vision import cv2
        mask = cv2.imread(str(check_pin(row['mask']['mask'], under=root)), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (480, 640) or not np.any(mask):
            raise ValueError('observed mask missing/empty')
        from .quality_adaptation_smoke import legacy_native_intrinsics
        native = legacy_native_intrinsics(np.asarray(manifest['intrinsics']))
        native.setflags(write=False)
        inputs[row['id']] = (bundle, manifest, native)
    return inputs


@dataclass(frozen=True, slots=True)
class _PreflightWitness:
    issuer: object
    raw_sha256: str
    context_sha256: str
    sink: object
    physical_sha256: str
    native_k: tuple


def observer_preflight(raw, expected_sha256, sink, *, context):
    from .quality_adaptation_refinement_states import RefStateCapture
    if type(sink) is not RefStateCapture:
        raise TypeError('exact retained state capture sink required')
    inputs = preflight_inputs(raw, expected_sha256, context)
    plan = _loads(raw); physical = next((r for r in plan['rows'] if r['id'] == sink.physical['id']), None)
    doc = context.document()
    if (physical is None or {k: v for k, v in sink.physical.items() if k != 'seed'} != physical or
            physical['id'] not in inputs or not np.array_equal(sink.native_k, inputs[physical['id']][2]) or
            Path(sink.output).resolve() != _capture_output(context) or
            sink.max_output_bytes != doc['limits']['storage_bytes'] or sink.reserve_bytes != doc['limits']['reserve_bytes']):
        raise ValueError('portable sink/physical/geometry mismatch')
    from .quality_adaptation_refinement_states import state_rows
    identities = ('id', 'object_id', 'split', 'sequence_id', 'capture_group', 'frame_id', 'timestamp_s', 'iteration')
    expected_states = state_rows(dict(rows=plan['rows']))
    if len(sink.rows) != len(expected_states) or any(
            any(row.get(key) != expected[key] for key in identities) for row, expected in zip(sink.rows, expected_states)):
        raise ValueError('portable sink requested inventory differs')
    return _PreflightWitness(_WITNESS_ISSUER, expected_sha256, context.sha256, sink,
        scale.canonical_hash(physical), tuple(map(tuple, sink.native_k)))


def validate_observer(capability, sink, geometry):
    from . import quality_adaptation_larger as larger
    from .quality_adaptation_refinement_states import RefStateCapture
    witness = capability._witness
    if (capability._issuer is not larger._PORTABLE_ISSUER or capability._policy != POLICY
            or type(capability._context) is not PortableContext or type(witness) is not _PreflightWitness
            or witness.issuer is not _WITNESS_ISSUER or type(sink) is not RefStateCapture
            or sink is not capability._sink or witness.sink is not sink or geometry != larger.inference.PROFILE
            or hashlib.sha256(capability._raw).hexdigest() != capability._sha256
            or witness.raw_sha256 != capability._sha256 or witness.context_sha256 != capability._context.sha256):
        raise ValueError('invalid portable observer capability/sink/profile')
    # Reauthenticate actual inputs/source and native geometry, not a caller's bool.
    current = observer_preflight(capability._raw, capability._sha256, sink, context=capability._context)
    if current.physical_sha256 != witness.physical_sha256 or current.native_k != witness.native_k:
        raise ValueError('portable observer binding changed')


def _capture_output(context):
    migration = _CAPTURE_SUCCESSOR.get()
    if migration is not None and migration[0] == context.sha256:
        return migration[2]
    return Path(context.document()['output_root']) / 'raw'


@contextmanager
def _capture_backend(context, raw, expected, inputs, states, output):
    """One physical-capture implementation, shared by both attempt kinds."""
    from . import quality_adaptation_larger as larger
    from . import quality_adaptation_refinement_states as shared
    from .quality_adaptation_larger_optimized import PreparedFoundPoseSearch, Timings
    from .quality_gotrack import load_network, TexturedRenderer, GoTrackRefiner
    from .quality_contract import Frame
    from .quality_time import PHYSICAL
    from .quality_runner import read_rgb
    from .vision import cv2
    doc = context.document(); root = Path(doc['output_root'])
    network = search = torch = None; current = None
    timings = Timings(doc['device']); execution_error = None
    try:
        torch = larger.configure_torch(doc['device'], deterministic=True); timings.torch = torch
        network = load_network(doc['device']).eval()
        def execute(physical):
            nonlocal search, current
            renderer = sink = refiner = seed_refiner = backend = None; reason = None
            if physical['split'] not in ('train', 'validation'):
                reason = 'sealed_' + physical['split'] + '_metadata_only'
                for row in states:
                    if row['id'].startswith(physical['id'] + ':iteration'):
                        row['failure_reason'] = reason
                return
            try:
                resource_check(context, 32 * 1024**2)
                if physical['id'] not in inputs:
                    raise ValueError('render_bank_or_observed_mask_unavailable')
                bundle, manifest, native_k = inputs[physical['id']]
                rgb = read_rgb(bundle, manifest, 0)
                mask = cv2.imread(str(check_pin(physical['mask']['mask'], under=root)), cv2.IMREAD_GRAYSCALE)
                frame = Frame(0, rgb, native_k, timestamp_s=0., clock_mode=PHYSICAL)
                random.seed(0); np.random.seed(0); torch.manual_seed(0)
                renderer = TexturedRenderer(bundle / 'object.glb', manifest['object_id'], unlit=True,
                    disable_multisampling=True, coordinate_mode='integer_centers_v1')
                seed_refiner = GoTrackRefiner(network, renderer, manifest['object_id'], doc['device'], inference_geometry=larger.inference.PROFILE)
                binding = dict(object_id=physical['object_id'], bank_pin=physical['bank']['bank'],
                    asset_sha256=physical['mesh_recipe']['sha256'], variant='amortized-v1')
                if current != physical['object_id']:
                    if search is not None:
                        search.close()
                    search = None
                    search = PreparedFoundPoseSearch(**binding, timings=timings,
                        loader=lambda path: torch.load(path, map_location='cpu', weights_only=False),
                        pin_resolver=lambda value: check_pin(value, under=root))
                    current = physical['object_id']
                backend = search.bind(seed_refiner, **binding, time_policy=manifest['_timing']['time_policy'])
                seeds = backend.recover(frame, mask, top_k=5)
                physical['seed'] = dict(status='available' if seeds else 'unavailable',
                    poses=[np.asarray(seed).tolist() for seed in seeds], selection='first_FoundPose_quality_sorted_hypothesis')
                if not seeds:
                    physical['seed']['failure_reason'] = 'no_estimated_FoundPose_hypothesis'
                    raise ValueError('unavailable estimated seed')
                backend = seed_refiner = None
                with shared.observe_selected_crops(physical['iterations']) as checks:
                    sink = shared.RefStateCapture(output, physical, states, checks, native_k,
                        max_output_bytes=doc['limits']['storage_bytes'], reserve_bytes=doc['limits']['reserve_bytes'],
                        resource_check=lambda: resource_check(context, 32 * 1024**2))
                    capability = larger.issue_portable_observer(raw, expected, sink, context=context)
                    refiner = GoTrackRefiner(network, renderer, manifest['object_id'], doc['device'],
                        inference_geometry=larger.inference.PROFILE, chronological_capture_callback=sink,
                        inference_observer_capability=capability)
                    refiner.set_chronological_capture_context(dict(physical_id=physical['id']))
                    refiner.refine(frame, mask, seeds[0])
            except ResourceError:
                raise
            except Exception as exc:
                reason = _failure(exc)
                if physical['seed']['status'] == 'pending':
                    physical['seed'] = dict(status='unavailable', failure_reason=reason)
            finally:
                if sink is not None:
                    sink.finish(reason)
                else:
                    for row in states:
                        if row['id'].startswith(physical['id'] + ':iteration'):
                            row['failure_reason'] = reason or 'capture_unavailable'
                if renderer is not None:
                    renderer.close()
                renderer = sink = refiner = seed_refiner = backend = None
        yield execute
    except BaseException as exc:
        execution_error = exc
        raise
    finally:
        cleanup = []
        for action in (lambda: search.close() if search is not None else None,
                lambda: network.cpu() if network is not None else None,
                lambda: larger.cleanup_device(doc['device'], torch) if torch is not None else None):
            try:
                action()
            except BaseException as exc:
                cleanup.append(_failure(exc))
        if cleanup:
            error = RuntimeError('raw capture cleanup failed: ' + '; '.join(cleanup))
            error.capture_cleanup_errors = cleanup
            error.capture_execution_error = execution_error
            raise error


def capture(context):
    _require_stage(context, 'capture')
    from . import quality_adaptation_refinement_states as shared
    doc, spec, scale_plan, physicals = frozen(context); root = Path(doc['output_root']); output = root / 'raw'
    output.mkdir(exist_ok=False)
    plan = capture_plan(context, physicals); raw = encoded(plan); expected = hashlib.sha256(raw).hexdigest()
    write_json(context, root / 'input-plan.json', plan)
    states = shared.state_rows(dict(rows=physicals))
    write_json(context, output / 'capture-index.json', dict(rows=states, training_ready=False))
    fatal = None; cleanup = []
    try:
        inputs = preflight_inputs(raw, expected, context)
        with _capture_backend(context, raw, expected, inputs, states, output) as execute:
            for physical in physicals:
                try:
                    execute(physical)
                finally:
                    if physical['split'] in ('train', 'validation'):
                        _save_rows(context, physicals)
                        write_json(context, output / 'capture-index.json', dict(rows=states, training_ready=False), replace=True)
    except BaseException as exc:
        if hasattr(exc, 'capture_cleanup_errors'):
            cleanup = exc.capture_cleanup_errors
            prior = exc.capture_execution_error
            fatal = None if prior is None else _failure(prior)
        else:
            fatal = _failure(exc)
        raise
    finally:
        for row in states:
            if row.get('failure_reason') == 'not_processed':
                row['failure_reason'] = fatal or 'unrun_after_termination'
        for physical in _active(physicals):
            if physical['seed']['status'] == 'pending':
                physical['seed'] = dict(status='unavailable', failure_reason=fatal or 'unrun_after_termination')
        _save_rows(context, physicals)
        write_json(context, output / 'capture-index.json', dict(rows=states, training_ready=False), replace=True)
        write_json(context, output / 'capture-receipt.json', dict(schema='kaggle_raw_capture_receipt_v1',
            input_plan_sha256=expected, context_sha256=context.sha256, fatal_error=fatal, cleanup_errors=cleanup,
            physical_frames_requested=len(physicals), refinement_states_requested=len(states),
            annotation_inputs=[], test_payload_reads=0, training_ready=False))
    return dict(requested=len(states), available=sum(r['status'] == 'available' for r in states))


def _resume_seconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid continuation ' + name)
    return value


def _owned_terminal_process(job, context, capture_elapsed):
    """Alternate witness for the retained original notebook Popen handle.

    The reviewed cell80 launcher records launch-capture.json and starts exactly
    this timeout/operator/request command. The actual Popen object's args and
    poll result can still be observed after /proc no longer has a start token.
    This route authenticates that complete chain, never a reconstructed token.
    """
    owner = job.get('popen_owner')
    _keys(owner, ('schema', 'handle_name', 'pid', 'returncode', 'args', 'args_sha256',
        'launch', 'request', 'operator', 'observation'), 'owned Popen terminal witness')
    args = owner['args']
    if (owner['schema'] != 'visualizeit_raw_owned_popen_terminal_v1'
            or owner['handle_name'] != 'expanded_group1_cached_phase_job'
            or owner['pid'] != job['pid'] or owner['returncode'] != job['returncode']
            or not isinstance(args, list) or len(args) != 11
            or any(not isinstance(v, str) or not v for v in args)
            or hashlib.sha256(encoded(args)).hexdigest() != owner['args_sha256']
            or args[:4] != ['timeout', '--signal=TERM', '--kill-after=15s', '7230s']
            or args[5:8] != ['-E', '-s', '-B']):
        raise ValueError('owned Popen PID/returncode/command identity differs')
    launch = read_json(check_pin(owner['launch'], max_bytes=MAX_METADATA))
    request_path = check_pin(owner['request'], max_bytes=MAX_METADATA)
    request = read_json(request_path)
    operator = check_pin(owner['operator'], max_bytes=MAX_METADATA)
    observation = read_json(check_pin(owner['observation'], max_bytes=MAX_METADATA))
    if (launch != dict(pid=job['pid'], stage='capture', request=owner['request'], training_executed=False)
            or request.get('schema') != 'visualizeit_expanded_cached_capture_stage_request_v1'
            or request.get('stage') != 'capture' or request.get('shard_id') != 'group1'
            or args[4] != request.get('python') or args[8] != operator.as_posix()
            or args[9] != request_path.as_posix() or args[10] != owner['request']['sha256']
            or observation.get('schema') != 'visualizeit_expanded_cached_capture_stage_observation_v1'
            or observation.get('stage') != 'capture' or observation.get('shard_id') != 'group1'
            or observation.get('context') != request.get('context')
            or observation.get('status') != 'incomplete' or observation.get('reason') != 'deadline_exceeded'
            or observation.get('returncode') != -15 or observation.get('training_executed') is not False
            or observation.get('test_payload_reads') != 0
            or observation.get('cleanup_errors', []) != [] or observation.get('log_errors', []) != []
            or _resume_seconds(observation.get('elapsed_seconds'), 'owned observation elapsed') != capture_elapsed):
        raise ValueError('owned Popen original launch/request/operator/observation chain differs')
    context_pin = request.get('context')
    if (not isinstance(context_pin, dict) or context_pin.get('sha256') != context.sha256
            or check_pin(context_pin, max_bytes=MAX_METADATA).read_bytes() != context.raw):
        raise ValueError('owned Popen original request context differs')


@contextmanager
def _successor_sources(context, successor, output):
    token = _CAPTURE_SUCCESSOR.set((context.sha256, successor, Path(output).resolve()))
    try:
        yield
    finally:
        _CAPTURE_SUCCESSOR.reset(token)


def authenticate_capture_continuation(context, successor, terminal_pin):
    """Authenticate one terminal predecessor; this API grants no execution.

    The caller supplies a pinned operator observation, not a stale-lock guess.
    Its ordered unit inventory must explicitly attest the unattempted suffix.
    Old source bytes are checked against archives for every changed file;
    executing bytes are checked against the separate successor context. Roots,
    recipe, inputs and all other context fields must be identical.
    """
    doc, next_doc = context.document(), successor.document()
    if {k: v for k, v in doc.items() if k != 'source_files'} != {
            k: v for k, v in next_doc.items() if k != 'source_files'}:
        raise ValueError('continuation context may change only explicit source pins')
    verify_sources(successor)
    root = Path(doc['output_root']); output = root / 'raw'
    terminal = read_json(check_pin(terminal_pin, max_bytes=MAX_METADATA))
    _keys(terminal, ('schema', 'context_sha256', 'metadata', 'source_archive',
        'job', 'units', 'capture_elapsed_seconds', 'prior_history'), 'terminal continuation evidence')
    if (terminal['schema'] != 'visualizeit_raw_capture_terminal_resume_v1'
            or terminal['context_sha256'] != context.sha256):
        raise ValueError('terminal predecessor context differs')
    job = terminal['job']
    job_keys = ('pid', 'returncode', 'process_start_token', 'process_start_token_at_poll',
        'terminal_observed_unix_s', 'absence_observed_unix_s', 'root_and_descendants_absent',
        'gpu_compute_pids', 'gpu_used_bytes')
    _keys(job, (*job_keys, 'popen_owner') if 'popen_owner' in job else job_keys, 'terminal process observation')
    if (type(job['pid']) is not int or job['pid'] <= 1
            or type(job['returncode']) is not int or job['returncode'] == 0
            or job['root_and_descendants_absent'] is not True
            or job['gpu_compute_pids'] != [] or job['gpu_used_bytes'] != 0
            or _resume_seconds(job['terminal_observed_unix_s'], 'terminal observation') <= 0
            or _resume_seconds(job['absence_observed_unix_s'], 'absence observation') < job['terminal_observed_unix_s']):
        raise ValueError('actual terminal process identity/absence and idle GPU observation required')
    if job['process_start_token'] is None and job['process_start_token_at_poll'] is None:
        _owned_terminal_process(job, context,
            _resume_seconds(terminal['capture_elapsed_seconds'], 'capture elapsed'))
    elif (not isinstance(job['process_start_token'], str) or not job['process_start_token']
            or job['process_start_token'] != job['process_start_token_at_poll'] or job.get('popen_owner') is not None):
        raise ValueError('actual terminal process start identity differs')
    changed = {name for name, digest in doc['source_files'].items()
               if next_doc['source_files'][name] != digest}
    if not isinstance(terminal['source_archive'], dict) or set(terminal['source_archive']) != changed:
        raise ValueError('exact changed predecessor source archive required')
    for name in changed:
        archived = terminal['source_archive'][name]
        if archived['sha256'] != doc['source_files'][name]:
            raise ValueError('predecessor source archive identity differs: ' + name)
        check_pin(archived)
    required = {'context.json', 'recipe.json', 'scale-plan.json', 'split.json',
        'freeze.json', 'physical-index.json', 'input-plan.json', 'raw/capture-index.json',
        'raw/capture-receipt.json', 'stage-capture.json'}
    # Bind every root metadata receipt, including cached-input provenance. A
    # null pin is accepted only for the two genuinely absent terminal receipts.
    required |= {path.name for path in root.glob('*.json')}
    pins = terminal['metadata']
    if not isinstance(pins, dict) or set(pins) != required:
        raise ValueError('complete exact predecessor metadata inventory required')
    for name in sorted(required):
        path = root / name
        if pins[name] is None:
            if name not in ('stage-capture.json', 'raw/capture-receipt.json') or path.exists() or path.is_symlink():
                raise ValueError('unobserved predecessor metadata absence: ' + name)
        elif check_pin(pins[name], under=root, max_bytes=MAX_METADATA) != path.resolve():
            raise ValueError('predecessor metadata path differs: ' + name)
    if sha256(root / 'context.json') != context.sha256:
        raise ValueError('predecessor context payload differs')
    with _successor_sources(context, successor, output):
        _, spec, plan, physicals = frozen(context)
        input_plan = read_json(root / 'input-plan.json')
        _validate_capture_plan(input_plan, context)
    states_doc = read_json(output / 'capture-index.json')
    _keys(states_doc, ('rows', 'training_ready'), 'predecessor capture index')
    if states_doc['training_ready'] is not False:
        raise ValueError('RAW predecessor cannot declare training readiness')
    states = states_doc['rows']
    from .quality_adaptation_refinement_states import state_rows
    identities = ('id', 'object_id', 'split', 'sequence_id', 'capture_group', 'frame_id', 'timestamp_s', 'iteration')
    expected = state_rows(dict(rows=physicals))
    if len(states) != len(expected) or any(any(row.get(k) != identity[k] for k in identities)
            for row, identity in zip(states, expected)):
        raise ValueError('missing/duplicate/reordered/changed retained state identity')
    units = terminal['units']
    if not isinstance(units, list) or len(units) != len(physicals):
        raise ValueError('complete ordered terminal physical unit inventory required')
    pending, preserved, artifacts = [], [], {}
    pending_started = False
    for physical, unit in zip(physicals, units):
        _keys(unit, ('id', 'outcome'), 'terminal physical unit')
        if unit['id'] != physical['id']:
            raise ValueError('missing/duplicate/reordered terminal physical identity')
        rows = [row for row in states if row['id'].startswith(physical['id'] + ':iteration')]
        if physical['split'] not in ('train', 'validation'):
            if unit['outcome'] != 'sealed' or any(row['status'] != 'unavailable'
                    or row.get('query_rgb_sha256') is not None or set(row) != set(expected[0])
                    for row in rows):
                raise ValueError('sealed TEST/dev capture payload forbidden')
            continue
        available = [row for row in rows if row['status'] == 'available']
        if available and len(available) != len(rows):
            raise ValueError('partial chronological physical unit is ambiguous; replay forbidden')
        if unit['outcome'] == 'unattempted':
            if (physical['seed'] != {'status': 'pending'} or any(
                    row != identity for row, identity in zip(rows,
                        [r for r in expected if r['id'].startswith(physical['id'] + ':iteration')]))):
                raise ValueError('unattempted unit lacks pristine chronology/seed evidence')
            pending_started = True
            pending.append(physical['id'])
        elif unit['outcome'] in ('completed', 'failed'):
            if pending_started:
                raise ValueError('terminal execution order is not an attempted prefix and unattempted suffix')
            if unit['outcome'] == 'completed':
                if len(available) != len(rows) or physical['seed']['status'] != 'available':
                    raise ValueError('completed physical unit lacks complete chronological capture')
                if any(not isinstance(row.get('pnp_state'), str) or not row['pnp_state']
                        or row['pnp_state'] == 'solver_result_not_observed' for row in rows):
                    raise ValueError('partial chronological solver observation is ambiguous; replay forbidden')
            elif (available or physical['seed']['status'] == 'pending' or any(
                    row['status'] != 'unavailable' or not row.get('failure_reason')
                    or row['failure_reason'] == 'not_processed' for row in rows)):
                raise ValueError('failed physical unit lacks explicit observed failure')
            preserved.append(physical['id'])
        else:
            raise ValueError('unknown terminal physical outcome')
    # All metadata and sealed identities have passed before payload traversal.
    for row in states:
        if row['status'] == 'available':
            require_digest(row.get('query_rgb_sha256'))
            for kind in ('crops', 'capture'):
                descriptor = row[kind]
                relative = PurePosixPath(descriptor['path'])
                if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 1:
                    raise ValueError('retained payload path must be one relative filename')
                value = dict(descriptor, path=(output / relative.name).as_posix())
                path = check_pin(value, under=output, max_bytes=16 * 1024**2)
                if path.as_posix() in artifacts:
                    raise ValueError('duplicate retained payload path')
                artifacts[path.as_posix()] = value
        elif row['status'] != 'unavailable' or row.get('crops') is not None or row.get('capture') is not None:
            raise ValueError('unavailable row has ambiguous retained payload')
    actual_files = {path.resolve().as_posix() for path in output.iterdir()
                    if path.name not in ('capture-index.json', 'capture-receipt.json')}
    if actual_files != set(artifacts):
        raise ValueError('orphan/partial/unindexed predecessor capture files; replay forbidden')
    receipt = None if pins['raw/capture-receipt.json'] is None else read_json(output / 'capture-receipt.json')
    if receipt is not None and (receipt.get('context_sha256') != context.sha256
            or receipt.get('input_plan_sha256') != pins['input-plan.json']['sha256']
            or receipt.get('annotation_inputs') != [] or receipt.get('test_payload_reads') != 0
            or receipt.get('training_ready') is not False or receipt.get('cleanup_errors') != []):
        raise ValueError('terminal capture receipt binding/cleanup differs')
    elapsed = _resume_seconds(read_json(root / 'freeze.json')['elapsed_seconds'], 'freeze elapsed')
    for name in sorted(required):
        if name.startswith('stage-') and pins[name] is not None:
            stage_doc = read_json(root / name)
            if stage_doc.get('context_sha256') != context.sha256:
                raise ValueError('predecessor stage context differs')
            elapsed += _resume_seconds(stage_doc['elapsed_seconds'], 'stage elapsed')
    capture_elapsed = _resume_seconds(terminal['capture_elapsed_seconds'], 'capture elapsed')
    if pins['stage-capture.json'] is not None:
        if capture_elapsed != read_json(root / 'stage-capture.json')['elapsed_seconds']:
            raise ValueError('capture stage and operator elapsed disagree; count only once')
    else:
        elapsed += capture_elapsed
    history = terminal['prior_history']
    historical = 0.
    phase_history = 'cached-capture-phase-validation-v1.json'
    if phase_history in pins and (history is None or history.get('receipt') != pins[phase_history]):
        raise ValueError('cached-phase prior history cannot be omitted or counted as a current stage')
    if history is not None:
        _keys(history, ('receipt', 'field'), 'prior history charge')
        history_path = check_pin(history['receipt'], under=root, max_bytes=MAX_METADATA)
        if (history_path.name != phase_history or pins.get(history_path.name) != history['receipt']
                or history['field'] != 'source_historical_elapsed_seconds'):
            raise ValueError('prior history must name the frozen cached-phase historical clock')
        historical = _resume_seconds(read_json(history_path)[history['field']], 'historical elapsed')
    return dict(schema='visualizeit_raw_capture_resume_authentication_v1', terminal=terminal_pin,
        predecessor_context_sha256=context.sha256, successor_context_sha256=successor.sha256,
        source_archive=terminal['source_archive'], metadata=pins, physicals=physicals, states=states,
        unattempted_physical_ids=pending, preserved_physical_ids=preserved, artifacts=artifacts,
        predecessor_phase_elapsed_seconds=elapsed, prior_history_elapsed_seconds=historical,
        cumulative_elapsed_seconds=historical + elapsed, original_phase_budget_seconds=doc['limits']['wall_seconds'],
        annotation_inputs=[], test_payload_reads=0, training_ready=False)


def _continuation_allowance(context, successor, terminal_pin, allowance_pin):
    allowance = read_json(check_pin(allowance_pin, max_bytes=MAX_METADATA))
    _keys(allowance, ('schema', 'operation', 'predecessor_context_sha256', 'successor_context_sha256',
        'terminal', 'additional_allowance', 'max_seconds', 'attempt_id'), 'continuation allowance')
    name = allowance['attempt_id']
    seconds = _resume_seconds(allowance['max_seconds'], 'allowance seconds')
    if (allowance['schema'] != 'visualizeit_raw_capture_continuation_allowance_v1'
            or allowance['operation'] != 'capture_continuation'
            or allowance['predecessor_context_sha256'] != context.sha256
            or allowance['successor_context_sha256'] != successor.sha256
            or allowance['terminal'] != terminal_pin or allowance['additional_allowance'] is not True
            or not 0 < seconds <= 7200 or not isinstance(name, str)
            or not name or len(name) > 64 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name)):
        raise ValueError('separate explicit bounded continuation allowance required')
    return allowance


@contextmanager
def _capture_continuation_guard(context, successor, output, seconds, *, cloud_execution):
    """The same cloud/resource boundaries as stage(), with a separate clock."""
    from . import quality_adaptation_larger as larger
    doc = context.document()
    if (not cloud_execution or platform.system() != 'Linux' or os.environ.get('VISUALIZEIT_CLOUD') != '1'
            or not Path('/kaggle/input').is_dir()
            or not Path(doc['output_root']).is_relative_to('/kaggle/working')
            or not Path(doc['scratch_root']).is_relative_to('/tmp')):
        raise RuntimeError('explicit Kaggle Linux cloud execution and scoped working/tmp roots required')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('continuation watchdog requires main thread')
    stopped = threading.Event(); problems = []
    def interrupt(signum, frame):
        raise ResourceError(problems[0] if problems else 'separate continuation deadline exhausted')
    def monitor():
        while not stopped.wait(.1):
            try:
                if _rss_tree() > doc['limits']['rss_bytes']:
                    raise ResourceError('process tree RSS bound exhausted')
                resource_check(context)
            except BaseException as exc:
                if not stopped.is_set():
                    problems.append(_failure(exc)); os.kill(os.getpid(), signal.SIGALRM)
                return
    with larger.stage_lock(), _successor_sources(context, successor, output):
        if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
            raise RuntimeError('another execution deadline is active')
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        try:
            resource_check(context)
            verify_neural_environment(context)
            yield
            resource_check(context)
        finally:
            stopped.set(); watcher.join(timeout=2)
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, previous)
            _close_neighbors()


def _require_fresh_capture_continuation(root):
    directory = _absolute((root / 'capture-continuations').as_posix(), 'continuation inventory')
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError('prior continuation exists for this predecessor; preserve it for explicit review')


def continue_capture(context, successor, terminal_pin, allowance_pin, *, cloud_execution=False):
    """Append a distinct capture attempt; never rewrite predecessor artifacts.

    This produces a continuation receipt/index, not a legacy RAW seal. Consumers
    must resolve each physical unit using artifact_roots and must not pass this
    mixed-attempt index to the original seal_capture() or training interfaces.
    A second interruption needs fresh terminal evidence and operator review;
    this API never replays partial units or automatically retries an attempt.
    """
    began = time.monotonic()
    allowance = _continuation_allowance(context, successor, terminal_pin, allowance_pin)
    root = Path(context.document()['output_root'])
    _require_fresh_capture_continuation(root)
    admission = authenticate_capture_continuation(context, successor, terminal_pin)
    if not admission['unattempted_physical_ids']:
        raise ValueError('no demonstrably unattempted physical work remains')
    attempt = _absolute((root / 'capture-continuations' / allowance['attempt_id']).as_posix(), 'continuation output')
    output = attempt / 'raw'
    if attempt.exists() or attempt.is_symlink():
        raise FileExistsError('continuation attempt already exists; preserve interrupted chronology')
    # Validation, guard setup, preflight and cleanup all spend the same separate
    # allowance. Earlier measured time is retained as history, never subtracted
    # from a new context or silently reset.
    remaining = allowance['max_seconds'] - (time.monotonic() - began)
    if remaining <= 0:
        raise ResourceError('continuation allowance exhausted during authentication')
    physicals, states = admission['physicals'], admission['states']
    eligible = set(admission['unattempted_physical_ids'])
    roots = {p['id']: (output if p['id'] in eligible else root / 'raw').as_posix() for p in physicals}
    attempted, completed = [], []; fatal = None; created = False
    def persist():
        write_json(context, attempt / 'physical-index.json', dict(rows=physicals), replace=True)
        write_json(context, attempt / 'capture-index.json', dict(rows=states, artifact_roots=roots,
            training_ready=False), replace=True)
    try:
        with _capture_continuation_guard(context, successor, output, remaining, cloud_execution=cloud_execution):
            _require_fresh_capture_continuation(root)
            # Authenticate once again after taking the real global stage lock.
            admission = authenticate_capture_continuation(context, successor, terminal_pin)
            attempt.mkdir(parents=True, exist_ok=False); created = True; output.mkdir()
            write_json(context, attempt / 'admission.json', dict(admission, allowance=allowance_pin))
            persist()
            raw = _read_metadata_bytes(root / 'input-plan.json')
            expected = admission['metadata']['input-plan.json']['sha256']
            inputs = preflight_inputs(raw, expected, context)
            with _capture_backend(context, raw, expected, inputs, states, output) as execute:
                for physical in physicals:
                    if physical['id'] not in eligible:
                        continue
                    attempted.append(physical['id'])
                    # Persist start before entering inference: interruption may
                    # leave a partial unit, and that unit must never look fresh.
                    write_json(context, attempt / 'progress.json', dict(attempted=attempted,
                        completed=completed, in_progress=physical['id']), replace=True)
                    try:
                        execute(physical)
                        completed.append(physical['id'])
                    finally:
                        persist()
                    write_json(context, attempt / 'progress.json', dict(attempted=attempted,
                        completed=completed, in_progress=None), replace=True)
    except BaseException as exc:
        fatal = _failure(exc)
        raise
    finally:
        if created:
            persist()
            # Rehash inherited bytes after cleanup. A callback cannot silently
            # corrupt predecessor payloads and still produce a successful receipt.
            preservation_error = None
            try:
                authenticate_capture_continuation(context, successor, terminal_pin)
            except BaseException as exc:
                preservation_error = _failure(exc)
            elapsed = time.monotonic() - began
            budget_exceeded = elapsed > allowance['max_seconds']
            receipt = dict(schema='visualizeit_raw_capture_continuation_receipt_v1',
                predecessor_terminal=terminal_pin, allowance=allowance_pin,
                predecessor_context_sha256=context.sha256, successor_context_sha256=successor.sha256,
                predecessor_metadata=admission['metadata'], source_archive=admission['source_archive'],
                physical_index=pin(attempt / 'physical-index.json'),
                capture_index=pin(attempt / 'capture-index.json'),
                progress=pin(attempt / 'progress.json') if (attempt / 'progress.json').exists() else None,
                status='failed' if fatal or preservation_error or budget_exceeded else 'completed', fatal_error=fatal,
                preservation_error=preservation_error, attempted_physical_ids=attempted,
                completed_physical_ids=completed, remaining_unattempted_physical_ids=[p for p in
                    admission['unattempted_physical_ids'] if p not in attempted],
                artifact_roots=roots, elapsed_seconds=elapsed,
                predecessor_phase_elapsed_seconds=admission['predecessor_phase_elapsed_seconds'],
                prior_history_elapsed_seconds=admission['prior_history_elapsed_seconds'],
                cumulative_elapsed_seconds=admission['cumulative_elapsed_seconds'] + elapsed,
                original_phase_budget_seconds=admission['original_phase_budget_seconds'],
                additional_allowance_seconds=allowance['max_seconds'],
                additional_allowance_exceeded=budget_exceeded,
                original_cumulative_budget_compliance=False, annotation_inputs=[], test_payload_reads=0,
                training_ready=False, legacy_raw_seal=False)
            write_json(context, attempt / 'receipt.json', receipt)
            if preservation_error:
                raise ValueError('predecessor preservation authentication failed: ' + preservation_error)
            if budget_exceeded:
                raise ResourceError('continuation allowance exhausted including final authentication/cleanup')
    return receipt


def seal_capture(context):
    """Seal successes AND failures; this never declares a trainable dataset."""
    _require_stage(context, 'seal-capture')
    doc, spec, plan, physicals = frozen(context); root = Path(doc['output_root']); output = root / 'raw'
    receipt = read_json(output / 'capture-receipt.json')
    states = read_json(output / 'capture-index.json')['rows']
    from .quality_adaptation_refinement_states import state_rows
    expected = state_rows(dict(rows=physicals))
    if [r['id'] for r in states] != [r['id'] for r in expected]:
        raise ValueError('raw state inventory incomplete/changed')
    if receipt['context_sha256'] != context.sha256 or sha256(root / 'input-plan.json') != receipt['input_plan_sha256']:
        raise ValueError('raw execution source/input binding changed')
    _validate_capture_plan(read_json(root / 'input-plan.json'), context)
    artifacts = {}
    for physical in physicals:
        if physical['split'] not in ('train', 'validation'):
            continue
        for kind, keys in (('render', ('input', 'video', 'mesh')), ('bank', ('bank', 'cnos_bank')),
                ('mask', ('mask', 'mask_index'))):
            if physical[kind]['status'] == 'available':
                for key in keys:
                    value = physical[kind][key]; check_pin(value, under=root)
                    artifacts[value['path']] = value
    for row, identity in zip(states, expected):
        if any(row.get(key) != identity[key] for key in ('object_id', 'split', 'sequence_id',
                'capture_group', 'frame_id', 'timestamp_s', 'iteration')):
            raise ValueError('raw state identity changed')
        if row['split'] in ('test', 'dev') and row['status'] != 'unavailable':
            raise ValueError('sealed state unexpectedly materialized')
        if row['status'] == 'available':
            for key in ('crops', 'capture'):
                descriptor = row[key]
                value = dict(descriptor, path=(output / descriptor['path']).as_posix())
                check_pin(value, under=output); artifacts[value['path']] = value
        elif row['status'] != 'unavailable' or not row.get('failure_reason') or row['failure_reason'] == 'not_processed':
            raise ValueError('raw state lacks explicit observed failure')
    for name in ('context.json', 'recipe.json', 'scale-plan.json', 'split.json', 'freeze.json',
            'physical-index.json', 'input-plan.json', 'raw/capture-index.json', 'raw/capture-receipt.json'):
        artifacts[name] = pin(root / name)
    for path in root.glob('stage-*.json'):
        artifacts[path.name] = pin(path)
    result = dict(schema='kaggle_raw_capture_seal_v1', context_sha256=context.sha256,
        prospective_plan_sha256=pin(root / 'scale-plan.json')['sha256'], artifacts=artifacts,
        physical_frames=physicals, refinement_states=states,
        physical_frames_requested=len(physicals), refinement_states_requested=len(states),
        available_crops=sum(r['status'] == 'available' for r in states),
        training_ready=False, prospective_inputs_are_recipe_identity_only=True,
        final_execution_inventory=False, annotation_inputs=[], test_payload_reads=0,
        capture_fatal_error=receipt['fatal_error'], cleanup_errors=receipt['cleanup_errors'])
    result['seal_sha256'] = scale.canonical_hash(result)
    write_json(context, root / 'raw-seal.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'render', 'banks', 'masks', 'capture', 'seal-capture'))
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--context-sha256', required=True)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--recipe-sha256')
    parser.add_argument('--cloud-execution', action='store_true')
    args = parser.parse_args(argv)
    context = read_context(args.context, args.context_sha256)
    if args.stage == 'freeze':
        if args.recipe is None or args.recipe_sha256 is None:
            parser.error('freeze requires --recipe and --recipe-sha256')
        # Freeze is metadata only and may run locally; it never opens asset payloads.
        result = freeze(args.recipe, args.recipe_sha256, context)
    else:
        with stage(context, args.stage, cloud_execution=args.cloud_execution):
            result = {'render': render, 'banks': banks, 'masks': masks,
                'capture': capture, 'seal-capture': seal_capture}[args.stage](context)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
