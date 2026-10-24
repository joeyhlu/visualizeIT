"""Additive, input-only controller for bounded multi-object GoTrack head adaptation.

Import is model-free. The immutable source-18 tools execute in an authenticated
separate checkout. The serving allowlist and historical experiment remain fixed.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
from . import quality_adaptation_refinement_states as shared
from . import quality_gotrack_inference as inference
from .quality_adaptation_data import (CONVENTION, CROP_KEYS, FEATURE_KEYS, HOOKS,
    SIZE, canonical_hash, crop_rgb_hash, descriptor, load_dataset, load_npz,
    read_json, require_hash, sha256, validate_crops, validate_features)
from .quality_adaptation_runtime import atomic_stream, cloud_guard, tree_bytes, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'gotrack_larger_input_plan_v1'
SOURCE_SCHEMA = 'gotrack_larger_source_contract_v1'
LABEL_SCHEMA = 'gotrack_larger_label_plan_v1'
SCRATCH = '/tmp/visualizeit-larger-v1'
DURABLE = '/teamspace/studios/this_studio/visualizeit-private-larger-v1'
HISTORICAL_MANIFEST_SHA256 = shared.HISTORICAL_MANIFEST_SHA256
HISTORICAL_LOCK_SHA256 = shared.HISTORICAL_LOCK_SHA256
LIMITS = dict(max_physical_frames=264, max_states=312, max_assets=13,
    max_aggregate_seconds=7200, max_process_tree_rss_bytes=8 * 1024**3,
    max_scratch_bytes=12 * 1024**3, max_new_durable_bytes=1_200_000_000,
    max_total_durable_bytes=9_500_000_000, existing_durable_bytes=7_716_719_791,
    disk_reserve_bytes=1024**3, model_cache_bytes=8 * 1024**3)
# Exact source admission; the parent freezes hashes after independent review.
SOURCE_FILES = (set(inference.EXPECTED_RUNTIME_FILES) | set(shared.EXTRA_SOURCES) |
    {'bench/quality_adaptation_larger.py', 'bench/quality_adaptation_larger_prepare.py',
     'bench/quality_adaptation_refinement_states.py', 'bench/quality_ranker_data.py'})
OPTIMIZED_SOURCE_FILES = SOURCE_FILES | {'bench/quality_adaptation_larger_optimized.py',
    'bench/quality_adaptation_larger_parity.py'}
PREPARATION_VARIANTS = ('legacy-v1', 'amortized-v1')


def expected_source_files(variant='legacy-v1'):
    preparation_variant(dict(preparation_variant=variant))
    return set(OPTIMIZED_SOURCE_FILES if variant == 'amortized-v1' else SOURCE_FILES)


def preparation_variant(doc, selected=None):
    variant = doc.get('preparation_variant', 'legacy-v1')
    if variant not in PREPARATION_VARIANTS or (selected is not None and selected != variant):
        raise ValueError('preparation variant differs from frozen input/source variant')
    return variant


def _variant_keys(doc, required, label):
    _keys(doc, tuple(required) + (('preparation_variant',) if 'preparation_variant' in doc else ()), label)
    preparation_variant(doc)


class CaptureTimings:
    def __init__(self, device='cpu', torch=None):
        self.device, self.torch, self.events = device, torch, []

    def sync(self):
        if self.device == 'cuda' and self.torch is not None:
            self.torch.cuda.synchronize()

    @contextmanager
    def step(self, name, *, view=None, temperature='unspecified'):
        self.sync(); began = time.perf_counter(); status = 'available'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            self.sync()
            self.events.append(dict(step=name, view=view, temperature=temperature,
                seconds=time.perf_counter() - began, status=status))

    def report(self):
        counts = {}
        for row in self.events:
            counts[row['step']] = counts.get(row['step'], 0) + 1
        return dict(events=self.events, observed_counts=counts,
            timing_overhead='synchronization and instrumentation included', cuda_synchronized=self.device == 'cuda',
            aggregation='events can be nested; do not sum as stage wall time',
            process_tree_rss_source='external stage supervision receipt',
            cuda_max_memory_allocated_bytes=(self.torch.cuda.max_memory_allocated()
                if self.device == 'cuda' and self.torch is not None else None))
_ISSUER = object()
_keys = shared._keys
_artifact = shared._artifact
_base = shared._base
GEOMETRY_KEYS = shared.GEOMETRY_KEYS


def _pin_shape(pin, *, source=False):
    _keys(pin, ('path', 'bytes', 'sha256'), 'file pin')
    require_hash(pin['sha256'])
    raw = pin['path']
    if (type(pin['bytes']) is not int or not 0 < pin['bytes'] <= LIMITS['max_scratch_bytes']
            or not isinstance(raw, str) or not raw or '\\' in raw or '\x00' in raw
            or '..' in PurePosixPath(raw).parts or str(PurePosixPath(raw)) != raw):
        raise ValueError('malformed bounded file pin')
    if source:
        if PurePosixPath(raw).is_absolute() or ':' in raw:
            raise ValueError('source pin must be relative POSIX')
    elif not (raw.startswith(SCRATCH + '/') or raw.startswith('/teamspace/studios/this_studio/')):
        raise ValueError('payload must stay in the approved scratch or studio roots')
    if any(token in raw.lower() for token in ('obj_000007', 'hot3d7', 'sealed-test')):
        raise ValueError('TEST payload is forbidden')
    return pin


def pinned_file(pin, *, source=False):
    _pin_shape(pin, source=source)
    path = shared._relative(ROOT, pin['path']) if source else Path(pin['path'])
    if inference._has_symlink_component(path):
        raise ValueError('pinned payload cannot contain symlinks')
    if not path.is_file() or path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
        raise ValueError('pinned payload changed: ' + pin['path'])
    return path


def _test_metadata(test):
    _keys(test, ('id', 'object_id', 'sequence_id', 'capture_group', 'split', 'status',
        'query_rgb_sha256', 'failure_reason', 'unavailable_provenance'), 'sealed TEST metadata')
    if (test['object_id'] != '7' or test['split'] != 'test' or test['status'] != 'unavailable'
            or test['query_rgb_sha256'] is not None or not test['failure_reason']
            or not test['unavailable_provenance']):
        raise ValueError('TEST7 must remain metadata only')


def validate_plan(plan, *, verify_files=False):
    _variant_keys(plan, ('schema', 'runtime_contract', 'source_contract', 'assets', 'rows', 'sealed_test', 'limits'), 'input plan')
    if plan['schema'] != SCHEMA or plan['limits'] != LIMITS:
        raise ValueError('larger plan schema/resource ceilings differ')
    for name in ('runtime_contract', 'source_contract'):
        _pin_shape(plan[name], source=True)
    _test_metadata(plan['sealed_test'])
    assets, ownership = {}, {}
    if not isinstance(plan['assets'], list) or not 2 <= len(plan['assets']) <= LIMITS['max_assets']:
        raise ValueError('plan requires 2..13 prospectively frozen assets')
    for asset in plan['assets']:
        _keys(asset, ('object_id', 'split', 'physical_object_id', 'original_mesh_sha256', 'mesh', 'bank'), 'asset')
        obj, split = asset['object_id'], asset['split']
        if (not isinstance(obj, str) or not obj.isdecimal() or str(int(obj)) != obj or obj == '7'
                or split not in ('train', 'val') or obj in assets
                or asset['physical_object_id'] != f'hot3d:obj_{int(obj):06d}'):
            raise ValueError('duplicate/invalid/TEST object identity')
        require_hash(asset['original_mesh_sha256'])
        _pin_shape(asset['mesh'])
        if asset['mesh']['sha256'] != asset['original_mesh_sha256']:
            raise ValueError('metric mesh must preserve original acquisition bytes')
        if asset['bank'] is not None:
            _pin_shape(asset['bank'])
        for identity in (asset['physical_object_id'], asset['original_mesh_sha256']):
            previous = ownership.setdefault(identity, split)
            if previous != split:
                raise ValueError('cross-split physical/mesh identity')
        assets[obj] = asset
    if set(a['split'] for a in assets.values()) != {'train', 'val'}:
        raise ValueError('both TRAIN and VAL asset owners required')
    rows = plan['rows']
    if not isinstance(rows, list) or not 2 <= len(rows) <= LIMITS['max_physical_frames']:
        raise ValueError('physical inventory is empty or exceeds bound')
    seen, groups, query_owners, count, asset_views = set(), {}, {}, 0, set()
    fields = {'id', 'object_id', 'split', 'sequence_id', 'capture_group', 'frame_id', 'timestamp_s',
              'view_index', 'input', 'video', 'mesh', 'mask_index', 'mask', 'bank', 'iterations',
              'observation_status', 'failure_reason'}
    for row in rows:
        _keys(row, fields, 'physical row')
        asset = assets.get(row['object_id'])
        if (asset is None or row['split'] != asset['split'] or type(row['view_index']) is not int
                or not 0 <= row['view_index'] < 24 or row['frame_id'] != 0 or type(row['frame_id']) is not int
                or type(row['timestamp_s']) not in (int, float) or row['timestamp_s'] != 0
                or row['id'] != row['sequence_id'] + ':frame0' or row['id'] in seen
                or not isinstance(row['sequence_id'], str) or not row['sequence_id']
                or not isinstance(row['capture_group'], str) or not 0 < len(row['capture_group']) <= 256):
            raise ValueError('invalid/duplicate normalized physical row')
        key = (row['object_id'], row['view_index'])
        if key in asset_views:
            raise ValueError('duplicate object/view')
        asset_views.add(key)
        seen.add(row['id'])
        for identity in (row['capture_group'], row['sequence_id']):
            if groups.setdefault(identity, row['split']) != row['split']:
                raise ValueError('cross-split capture/sequence identity')
        selected = row['iterations']
        if (not isinstance(selected, list) or not selected or selected != sorted(set(selected))
                or any(type(i) is not int or i not in (0, 1, 3) for i in selected)):
            raise ValueError('invalid/empty selected refinement iterations')
        count += len(selected)
        if (row['mesh'] is None or row['mesh']['sha256'] != asset['mesh']['sha256'] or row['mesh']['bytes'] != asset['mesh']['bytes']) or row['bank'] != asset['bank']:
            raise ValueError('row mesh/bank differs from frozen asset authorization')
        if row['observation_status'] not in ('available', 'unavailable'):
            raise ValueError('invalid prospective observation status')
        if row['observation_status'] == 'unavailable' and not row['failure_reason']:
            raise ValueError('unavailable observation requires retained failure')
        for name in ('input', 'video', 'mesh', 'mask_index', 'mask', 'bank'):
            pin = row[name]
            if pin is None:
                if row['observation_status'] == 'available':
                    raise ValueError('available observation lacks mask/bank/input')
                continue
            _pin_shape(pin)
            if any(token in pin['path'].lower() for token in ('reference', 'evaluation', 'generation', 'poses.json')):
                raise ValueError('annotation payload in input-only plan')
        if row['input'] is not None:
            bundle = PurePosixPath(row['input']['path']).parent
            if (PurePosixPath(row['input']['path']).name != 'input.json'
                    or row['video'] is None or row['video']['path'] != str(bundle / 'synthetic.mp4')
                    or row['mesh']['path'] != str(bundle / 'object.glb')):
                # Each rendered bundle owns an exact byte copy of the original mesh.
                raise ValueError('input/video/mesh must bind one normalized bundle')
            identity = row['video']['sha256']
            if query_owners.setdefault(identity, row['split']) != row['split']:
                raise ValueError('cross-split query RGB/video hash')
    if not 2 <= count <= LIMITS['max_states']:
        raise ValueError('state inventory empty or exceeds bound')
    # Complete lexical authorization precedes the FIRST payload file open.
    if verify_files:
        for name in ('runtime_contract', 'source_contract'):
            pinned_file(plan[name], source=True)
        for row in rows:
            for name in ('input', 'video', 'mesh', 'mask_index', 'mask', 'bank'):
                if row[name] is not None:
                    pinned_file(row[name])
    return plan


def verify_source_contract(plan, *, upstream=True):
    contract = read_json(pinned_file(plan['source_contract'], source=True))
    _variant_keys(contract, ('schema', 'sources', 'historical_training_manifest_sha256'), 'source closure')
    variant = preparation_variant(plan)
    preparation_variant(contract, variant)
    if contract['schema'] != SOURCE_SCHEMA or contract['historical_training_manifest_sha256'] != HISTORICAL_MANIFEST_SHA256:
        raise ValueError('wrong source closure/historical trainer')
    sources = {}
    for pin in contract['sources']:
        _keys(pin, ('path', 'sha256'), 'source hash')
        if pin['path'] in sources:
            raise ValueError('duplicate source hash')
        if sha256(shared._relative(ROOT, pin['path'])) != require_hash(pin['sha256']):
            raise ValueError('source closure mutation: ' + pin['path'])
        sources[pin['path']] = pin['sha256']
    if set(sources) != expected_source_files(variant):
        raise ValueError('source closure missing/foreign dependency')
    runtime = read_json(pinned_file(plan['runtime_contract'], source=True))
    inference._validate_contract(runtime)  # Preserve old fixed serving contract.
    if upstream:
        verify_upstream_sources()
    return runtime


def verify_upstream_sources():
    """Original lock pins authenticate immutable upstream trees, separately from wrappers."""
    from .quality_adaptation_runtime import LOCK
    if sha256(LOCK) != HISTORICAL_LOCK_SHA256:
        raise ValueError('historical source lock mutation')
    lock = read_json(LOCK)
    cache = (ROOT / '.cache/model-quality').resolve(strict=True)
    if os.name != 'nt' and cache.as_posix() != inference.APPROVED_CACHE_MOUNT:
        raise ValueError('upstream cache mount differs from approved cloud mount')
    for kind in ('gotrack', 'sam2'):
        for relative, expected in lock[kind]['files'].items():
            if not relative.startswith('.cache/model-quality/'):
                continue  # New wrappers have their separate complete source closure.
            path = (ROOT / relative).resolve(strict=True)
            if not path.is_relative_to(cache) or not path.is_file() or sha256(path) != require_hash(expected):
                raise ValueError('immutable upstream source mutation: ' + relative)


def read_plan(path, expected):
    raw = Path(path).read_bytes()
    if len(raw) > 4 * 1024**2 or hashlib.sha256(raw).hexdigest() != require_hash(expected):
        raise ValueError('frozen input plan byte hash/size mismatch')
    return raw, validate_plan(inference._loads_unique(raw, 'larger input plan'))


def state_rows(plan):
    return shared.state_rows(plan)


@dataclass(frozen=True, slots=True)
class LargerObserverCapability:
    _issuer: object
    _raw: bytes
    _sha256: str
    _sink: object
    _policy: str = 'legacy-v1'
    _context: object = None
    _witness: object = None

    def validate(self, sink, geometry):
        if self._policy == 'kaggle-activation-v1':
            from .quality_kaggle_activation import validate_observer
            return validate_observer(self, sink, geometry)
        if self._policy == 'portable-raw-v1':
            from .quality_adaptation_portable import validate_observer
            return validate_observer(self, sink, geometry)
        if self._policy != 'legacy-v1' or self._context is not None or self._witness is not None:
            raise ValueError('unknown/mixed larger observer policy')
        if (self._issuer is not _ISSUER or type(sink) is not shared.RefStateCapture
                or sink is not self._sink or geometry != inference.PROFILE
                or hashlib.sha256(self._raw).hexdigest() != self._sha256):
            raise ValueError('invalid larger observer capability/sink/profile')
        verify_source_contract(validate_plan(inference._loads_unique(self._raw, 'larger plan')), upstream=False)


def _issue(raw, expected, sink, verified):
    if verified != (_ISSUER, hashlib.sha256(raw).hexdigest()) or expected != verified[1]:
        raise ValueError('observer requires complete input-only preflight')
    cap = LargerObserverCapability(_ISSUER, raw, expected, sink)
    cap.validate(sink, inference.PROFILE)
    return cap


_PORTABLE_ISSUER = object()


def issue_portable_observer(raw, expected_sha256, sink, *, context):
    """Issue only after genuine portable input-only preflight, never a receipt flag."""
    from .quality_adaptation_portable import observer_preflight
    witness = observer_preflight(raw, expected_sha256, sink, context=context)
    cap = LargerObserverCapability(_PORTABLE_ISSUER, raw, expected_sha256, sink,
        'portable-raw-v1', context, witness)
    cap.validate(sink, inference.PROFILE)
    return cap


@contextmanager
def stage_lock():
    # Share the old stage lock so neural residency cannot overlap old workflows.
    with shared._stage_lock():
        yield


def _input_context(runtime, row, manifest):
    entry = dict(physical_object_id=f"hot3d:obj_{int(row['object_id']):06d}",
        native_resolution=[640, 480], input_sha256=row['input']['sha256'],
        mesh_path='object.glb', mesh_sha256=row['mesh']['sha256'],
        video_path='synthetic.mp4', video_sha256=row['video']['sha256'])
    if (manifest.get('schema_version') != 2 or manifest.get('provenance_kind') != 'synthetic'
            or manifest.get('units') != 'metres' or manifest.get('asset') != 'object.glb'
            or manifest.get('video') != 'synthetic.mp4' or manifest.get('_capture') is not None
            or manifest.get('physical_object_id') != entry['physical_object_id']
            or manifest.get('sequence_id') != row['sequence_id']
            or manifest.get('native_resolution') != entry['native_resolution']
            or manifest.get('source_hashes') != {'object.glb': entry['mesh_sha256'], 'synthetic.mp4': entry['video_sha256']}
            or manifest.get('frame_ids') != [0] or manifest['_timing']['clock_mode'] != 'capture_seconds_v1'):
        raise ValueError('observed manifest differs from frozen input-only authorization')
    original = inference._validated_native_intrinsics(manifest, entry)
    effective = np.array(original, dtype=np.float64, copy=True)
    effective[0, 2] += inference.NATIVE_CORRECTION[0]
    effective[1, 2] += inference.NATIVE_CORRECTION[1]
    effective.setflags(write=False)
    return shared._InputContext(inference.PROFILE, inference._freeze(runtime),
        inference._freeze(dict(inference._artifact_for_contract(runtime))), inference._freeze(entry), effective)


def validate_observed_mask(row, masks, plan):
    expected = dict(schema='gotrack_larger_observed_sam_mask_v1', frame_id=0,
        input_sha256=row['input']['sha256'], mesh_sha256=row['mesh']['sha256'],
        video_sha256=row['video']['sha256'], source_contract_sha256=plan['source_contract']['sha256'],
        producer_sha256=sha256(ROOT / ('bench/quality_adaptation_larger_optimized.py'
            if preparation_variant(plan) == 'amortized-v1' else 'bench/quality_adaptation_larger_prepare.py')),
        automatic=True, mask_origin='observed_CNOS_box_then_SAM2_image_predictor',
        mask=row['mask'], test_payload_reads=0, annotation_inputs=[])
    if any(masks.get(k) != value for k, value in expected.items()):
        raise ValueError('observed automatic SAM/CNOS producer binding differs')
    preparation_variant(masks, preparation_variant(plan))
    larger_pin = masks.get('cnos_bank')
    _pin_shape(larger_pin)
    pinned_file(larger_pin)
    box = masks.get('prompt_box_xyxy')
    if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
            or not (0 <= box[0] <= box[2] < 640 and 0 <= box[1] <= box[3] < 480)):
        raise ValueError('invalid observed CNOS prompt box')
    detection = masks.get('detector')
    if not isinstance(detection, dict) or str(detection.get('bank_sha256', '')).lower() != larger_pin['sha256']:
        raise ValueError('CNOS association receipt differs from scratch bank')


def preflight(plan):
    validate_plan(plan)  # Full inventory, including LAST row, before any payload.
    runtime = verify_source_contract(plan)
    validate_plan(plan, verify_files=True)
    from .quality_runner import read_input
    inputs = {}
    for row in plan['rows']:
        if row['observation_status'] != 'available':
            continue
        bundle = pinned_file(row['input']).parent
        manifest = read_input(bundle)
        context = _input_context(runtime, row, manifest)
        index = pinned_file(row['mask_index'])
        validate_observed_mask(row, read_json(index), plan)
        inputs[row['id']] = (bundle, manifest, context)
    return inputs


def _safe_output(path, *, durable=False):
    path = Path(path)
    prefix = DURABLE + '/' if durable else SCRATCH + '/'
    if ('..' in path.parts or not (path.as_posix().startswith(prefix) or durable and path.as_posix() == DURABLE)
            or inference._has_symlink_component(path)):
        raise ValueError('stage output escapes approved scratch/durable root')
    return path


def resources(scratch, durable=None, *, reserve=0):
    scratch = _safe_output(scratch)
    anchor = scratch
    while not anchor.exists():
        anchor = anchor.parent
    durable_root = Path('/teamspace/studios/this_studio')
    if durable_root.exists() and anchor.stat().st_dev == durable_root.stat().st_dev:
        raise ValueError('scratch and studio must be distinct filesystem mounts')
    if tree_bytes(Path(SCRATCH)) + reserve > LIMITS['max_scratch_bytes']:
        raise ValueError('scratch byte ceiling including atomic temporary exceeded')
    if shutil.disk_usage(anchor).free < reserve + LIMITS['disk_reserve_bytes']:
        raise ValueError('scratch reserve below one GiB')
    if tree_bytes(ROOT / '.cache/model-quality') > LIMITS['model_cache_bytes']:
        raise ValueError('existing model cache exceeds eight GiB; no eviction')
    if durable is not None:
        durable = _safe_output(durable, durable=True)
        budget_root = Path(DURABLE)
        used = tree_bytes(budget_root)
        if used + reserve > LIMITS['max_new_durable_bytes'] or LIMITS['existing_durable_bytes'] + used + reserve > LIMITS['max_total_durable_bytes']:
            raise ValueError('new/total durable peak ceiling exceeded')
        parent = durable
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < reserve + LIMITS['disk_reserve_bytes']:
            raise ValueError('durable reserve below one GiB')


def remaining(root, durable=None):
    receipt = read_json(Path(root) / 'capture-receipt.json')
    started = receipt.get('started_unix_s')
    if type(started) not in (int, float) or not np.isfinite(started) or started > time.time() + 1:
        raise ValueError('invalid original aggregate clock')
    left = LIMITS['max_aggregate_seconds'] - (time.time() - started)
    if left <= 0:
        raise TimeoutError('7200-second aggregate budget exhausted; no automatic reset')
    resources(root, durable)
    return left


def validate_device(device):
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be explicitly cpu or cuda')
    return device


def configure_torch(device, torch_module=None, *, deterministic=None):
    """Select a requested device before constructing any model; never fall back."""
    validate_device(device)
    if device == 'cuda':
        configured = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if configured != ':4096:8':
            raise ValueError('CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Torch import')
    if torch_module is None:
        import torch as torch_module
    if device == 'cuda' and not torch_module.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested but unavailable; no CPU fallback')
    torch_module.set_num_threads(1)
    if deterministic is not None or device == 'cuda':
        torch_module.use_deterministic_algorithms(True if deterministic is None else deterministic)
    if device == 'cuda':
        torch_module.backends.cuda.matmul.allow_tf32 = False
        torch_module.backends.cudnn.allow_tf32 = False
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True
    return torch_module


def cleanup_device(device, torch_module=None):
    if device == 'cuda':
        import gc
        gc.collect()
        if torch_module is None:
            import torch as torch_module
        torch_module.cuda.synchronize()
        torch_module.cuda.empty_cache()


def capture_device(root):
    return validate_device(read_json(Path(root) / 'capture-receipt.json').get('device', 'cpu'))


_CUDA_BOOTSTRAP = r"""
import os, runpy, sys
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    raise RuntimeError('CUDA deterministic workspace was not configured before import')
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA explicitly requested but unavailable; no CPU fallback')
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
try:
    runpy.run_module(module, run_name='__main__', alter_sys=True)
finally:
    import gc, json
    from pathlib import Path
    target = Path(os.environ['VISUALIZEIT_CUDA_RUNTIME_RECEIPT'])
    if target.parent.is_dir():
        evidence = dict(schema='gotrack_larger_cuda_runtime_v1', device='cuda',
            bootstrap_sha256=os.environ['VISUALIZEIT_CUDA_BOOTSTRAP_SHA256'],
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            threads=torch.get_num_threads())
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(evidence, sort_keys=True), encoding='utf-8')
        os.replace(temporary, target)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
"""


def run_historical_device(root, manifest, module, args, *, device='cpu', **kwargs):
    validate_device(device)
    if device == 'cpu':
        return shared._run_historical(root, manifest, module, args, **kwargs)
    shared.verify_training_root(root, manifest)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    options = list(map(str, args))
    is_export = module == 'bench.quality_adaptation_train' and options[0] == 'export'
    output = Path(options[options.index('--run' if is_export else '--output') + 1])
    env['VISUALIZEIT_CUDA_RUNTIME_RECEIPT'] = str(output / ('export-device-runtime.json' if is_export else 'device-runtime.json'))
    env['VISUALIZEIT_CUDA_BOOTSTRAP_SHA256'] = hashlib.sha256(_CUDA_BOOTSTRAP.encode()).hexdigest()
    accepted = kwargs.pop('accepted_exit_codes', (0,))
    completed = subprocess.run([sys.executable, '-B', '-c', _CUDA_BOOTSTRAP, module, *map(str, args)],
        cwd=Path(root).resolve(), env=env, check=False, **kwargs)
    shared.verify_training_root(root, manifest)
    if completed.returncode not in accepted:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return completed.returncode


def capture(plan_path, plan_sha256, output, *, cloud_execution=False, device='cpu', variant='legacy-v1'):
    cloud_guard(cloud_execution)
    validate_device(device)
    raw, plan = read_plan(plan_path, plan_sha256)
    preparation_variant(plan, variant)
    output = _safe_output(output)
    resources(output)
    with stage_lock():
        output.mkdir(parents=True, exist_ok=False)
        (output / 'plan.json').write_bytes(raw)
        rows = state_rows(plan)
        write_json(output, 'capture-index.json', dict(rows=rows, test_payload_reads=0))
        started, clock = time.monotonic(), float(os.environ.get('VISUALIZEIT_LARGER_CUDA_STARTED' if device == 'cuda' else 'VISUALIZEIT_LARGER_STARTED', time.time()))
        network, fatal, versions, torch = None, None, None, None
        search_cache = None
        timings = CaptureTimings(device)
        candidates = []
        try:
            with timings.step('preflight_hash_setup', temperature='cold'):
                inputs = preflight(plan)
            verified = (_ISSUER, plan_sha256)
            from .quality_adaptation_runtime import runtime_versions
            versions = runtime_versions()
            torch = configure_torch(device, deterministic=True)
            timings.torch = torch
            from .quality_gotrack import GoTrackRefiner, TexturedRenderer, load_network
            from .quality_foundpose import FoundPoseRecovery
            from .quality_contract import Frame
            from .quality_runner import read_rgb
            from .quality_time import PHYSICAL
            from .vision import cv2
            with timings.step('network_load', temperature='cold'):
                network = load_network(device).eval()
            if variant == 'amortized-v1':
                from .quality_adaptation_larger_optimized import ObjectSearchCache
                search_cache = ObjectSearchCache()
            for physical in plan['rows']:
                renderer = sink = bank = backend = seed_refiner = refiner = None
                reason = physical['failure_reason']
                try:
                    resources(output)
                    if time.time() - clock >= LIMITS['max_aggregate_seconds']:
                        raise TimeoutError('aggregate capture budget exhausted')
                    if physical['observation_status'] != 'available':
                        raise ValueError('prepared_observation_unavailable: ' + reason)
                    bundle, manifest, context = inputs[physical['id']]
                    rgb = read_rgb(bundle, manifest, physical['frame_id'])
                    mask = cv2.imread(str(pinned_file(physical['mask'])), cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape != rgb.shape[:2] or not np.any(mask):
                        raise ValueError('observed automatic mask missing/empty/wrong shape')
                    frame = Frame(0, rgb, context.effective_intrinsics, timestamp_s=0., clock_mode=PHYSICAL)
                    random.seed(0); np.random.seed(0); torch.manual_seed(0)
                    with timings.step('renderer_setup', view=physical['id'], temperature='cold'):
                        renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'],
                            unlit=True, disable_multisampling=True, coordinate_mode='integer_centers_v1')
                    seed_refiner = GoTrackRefiner(network, renderer, manifest['object_id'], device, inference_geometry=inference.PROFILE)
                    if search_cache is not None:
                        binding = dict(object_id=physical['object_id'], bank_pin=physical['bank'],
                            asset_sha256=physical['mesh']['sha256'], variant=variant)
                        search = search_cache.get(**binding, timings=timings,
                            loader=lambda path: torch.load(path, map_location='cpu', weights_only=False))
                        with timings.step('search_bind', view=physical['id']):
                            backend = search.bind(seed_refiner, **binding, time_policy=manifest['_timing']['time_policy'])
                    else:
                        with timings.step('bank_hash', view=physical['id'], temperature='cold'):
                            bank_path = pinned_file(physical['bank'])
                        with timings.step('bank_load', view=physical['id'], temperature='cold'):
                            bank = torch.load(bank_path, map_location='cpu', weights_only=False)
                        if bank.get('template_count') != 798:
                            raise ValueError('standard 798-template FoundPose bank required')
                        with timings.step('search_indices', view=physical['id'], temperature='cold'):
                            backend = FoundPoseRecovery(seed_refiner, bank, time_policy=manifest['_timing']['time_policy'])
                    with timings.step('recovery', view=physical['id']):
                        seeds = backend.recover(frame, mask, top_k=5)
                    candidates.append(dict(id=physical['id'], poses=[np.asarray(seed).tolist() for seed in seeds]))
                    bank = backend = seed_refiner = None
                    if not seeds:
                        raise ValueError('unavailable_seed: no estimated FoundPose hypothesis')
                    with shared.observe_selected_crops(physical['iterations']) as checks:
                        sink = shared.RefStateCapture(output, physical, rows, checks, context.effective_intrinsics)
                        capability = _issue(raw, plan_sha256, sink, verified)
                        refiner = GoTrackRefiner(network, renderer, manifest['object_id'], device,
                            inference_geometry=inference.PROFILE, chronological_capture_callback=sink,
                            inference_observer_capability=capability)
                        refiner.set_chronological_capture_context(dict(physical_id=physical['id']))
                        with timings.step('refine', view=physical['id']):
                            refiner.refine(frame, mask, seeds[0])
                except Exception as exc:
                    reason = f'{type(exc).__name__}: {exc}'[:1024]
                finally:
                    if sink is not None:
                        sink.finish(reason)
                    else:
                        for row in rows:
                            if row['id'].startswith(physical['id'] + ':iteration'):
                                row['failure_reason'] = reason or 'capture_unavailable'
                    if renderer is not None:
                        try:
                            with timings.step('renderer_release', view=physical['id']):
                                renderer.close()
                        except Exception as exc:
                            fatal = f'renderer cleanup failed: {exc}'[:1024]
                    refiner = bank = backend = seed_refiner = renderer = sink = None
                    with timings.step('inventory_write', view=physical['id']):
                        write_json(output, 'capture-index.json', dict(rows=rows, test_payload_reads=0))
                    if fatal:
                        break
        except Exception as exc:
            fatal = f'{type(exc).__name__}: {exc}'[:1024]
        finally:
            if search_cache is not None:
                try:
                    search_cache.close()
                except Exception as exc:
                    fatal = f'search cleanup failed: {exc}'[:1024]
            if network is not None:
                try:
                    network.cpu()
                except Exception as exc:
                    fatal = f'network cleanup failed: {exc}'[:1024]
                network = None
            if torch is not None:
                try:
                    with timings.step('device_release'):
                        cleanup_device(device, torch)
                except Exception as exc:
                    fatal = f'device cleanup failed: {exc}'[:1024]
            for row in rows:
                if row.get('failure_reason') == 'not_processed':
                    row['failure_reason'] = fatal or 'unrun_after_termination'
            write_json(output, 'capture-index.json', dict(rows=rows, test_payload_reads=0))
            write_json(output, 'capture-receipt.json', dict(schema='gotrack_larger_capture_receipt_v1',
                plan_sha256=plan_sha256, rows=rows, versions=versions, annotation_inputs=[], test_payload_reads=0,
                elapsed_seconds=time.monotonic() - started, fatal_error=fatal, started_unix_s=clock,
                input_only=True, device=device, execution_variant=device + '-v1', preparation_variant=variant, deterministic_algorithms=True,
                tf32=False if device == 'cuda' else None, physical_frames_requested=len(plan['rows']), refinement_states_requested=len(rows)))
            write_json(output, 'crops.json', dict(schema='correspondence_input_crops_v1', conventions=CONVENTION,
                rows=[dict(id=r['id'], crops=r['crops']) for r in rows if r['status'] == 'available']))
            write_json(output, 'preparation-diagnostics.json', dict(preparation_variant=variant,
                candidates=candidates, timings=timings.report(), test_payload_reads=0, annotation_inputs=[]))
        if fatal:
            raise RuntimeError('capture failed; complete requested inventory retained: ' + fatal)
        return dict(requested=len(rows), available=sum(r['status'] == 'available' for r in rows), test_payload_reads=0)


def read_capture(root):
    root = Path(root)
    plan = validate_plan(read_json(root / 'plan.json'))
    verify_source_contract(plan)
    index, receipt = read_json(root / 'capture-index.json'), read_json(root / 'capture-receipt.json')
    preparation_variant(receipt, preparation_variant(plan))
    expected, rows = state_rows(plan), index.get('rows')
    if (not isinstance(rows, list) or len(rows) != len(expected) or receipt.get('rows') != rows
            or receipt.get('plan_sha256') != sha256(root / 'plan.json') or receipt.get('fatal_error')
            or receipt.get('annotation_inputs') != [] or receipt.get('input_only') is not True
            or receipt.get('test_payload_reads') != 0 or index.get('test_payload_reads') != 0):
        raise ValueError('capture receipt/inventory/plan mutation')
    query_owners = {}
    for actual, prospect in zip(rows, expected):
        if (any(actual.get(k) != prospect[k] for k in ('id', 'object_id', 'split', 'sequence_id', 'capture_group', 'frame_id', 'timestamp_s', 'iteration'))
                or actual.get('status') not in ('available', 'unavailable')):
            raise ValueError('capture row differs from prospective physical state')
        digest = actual.get('query_rgb_sha256')
        if actual['status'] == 'available':
            require_hash(digest)
            if query_owners.setdefault(digest, actual['split']) != actual['split']:
                raise ValueError('cross-split retained query RGB hash')
        elif not actual.get('failure_reason'):
            raise ValueError('unavailable capture must retain failure reason')
    return plan, rows


def extract_stage(root, features, training_source_root, training_source_manifest, *, cloud_execution=False, device='cpu'):
    cloud_guard(cloud_execution)
    root, features = Path(root), _safe_output(features)
    _, rows = read_capture(root)
    if validate_device(device) != capture_device(root):
        raise ValueError('extraction device differs from authenticated capture device')
    with stage_lock():
        left = remaining(root)
        if features.exists():
            raise FileExistsError('extraction output must be new; failed trials stay preserved')
        try:
            return run_historical_device(training_source_root, training_source_manifest,
                'bench.quality_adaptation_prepare', ['--manifest', (root / 'crops.json').resolve(),
                    '--output', features.resolve(), '--device', device, '--max-seconds', max(1, int(left)),
                    '--max-output-bytes', LIMITS['model_cache_bytes'], '--max-rows', len(rows), '--cloud-execution'], timeout=left, device=device)
        finally:
            features.mkdir(parents=True, exist_ok=True)
            path = features / 'index.json'
            index = read_json(path) if path.is_file() else {}
            available = {r['id']: r for r in index.get('rows', [])}
            if set(available) - {r['id'] for r in rows}:
                raise ValueError('extractor contains foreign state')
            index['rows'] = [available.get(r['id'], dict(id=r['id'], status='unavailable',
                failure_reason=r.get('failure_reason') or 'extraction_failure_or_not_processed')) for r in rows]
            write_json(features, 'index.json', index, max_output_bytes=LIMITS['max_scratch_bytes'])
            remaining(root)


def _seal_material(root, features):
    plan, rows = read_capture(root)
    crops_doc, recipe, index = (read_json(root / 'crops.json'), read_json(features / 'recipe.json'), read_json(features / 'index.json'))
    expected_crops = dict(schema='correspondence_input_crops_v1', conventions=CONVENTION,
        rows=[dict(id=r['id'], crops=r['crops']) for r in rows if r['status'] == 'available'])
    if crops_doc != expected_crops:
        raise ValueError('input-only crop manifest differs from capture inventory')
    from .quality_adaptation_runtime import LOCK
    lock = read_json(LOCK)
    if (sha256(LOCK) != HISTORICAL_LOCK_SHA256
            or recipe.get('schema') != 'frozen_gotrack_decoder_recipe_v1'
            or recipe.get('source_lock_sha256') != HISTORICAL_LOCK_SHA256
            or recipe.get('source_files_sha256') != canonical_hash(lock['gotrack']['files'])
            or recipe.get('upstream_revision') != lock['gotrack']['revision']
            or recipe.get('producer_sha256') != lock['gotrack']['files']['bench/quality_adaptation_prepare.py']
            or recipe.get('hooks') != list(HOOKS) or recipe.get('annotation_inputs') != []
            or recipe.get('crop_manifest_sha256') != canonical_hash(crops_doc)
            or recipe.get('rgb_encoding') != 'uint8_div255_or_exact_float32_0_1_v1'
            or index.get('recipe_sha256') != canonical_hash(recipe)
            or [r.get('id') for r in index.get('rows', [])] != [r['id'] for r in rows]):
        raise ValueError('historical extractor recipe/index/source binding differs')
    execution_device = capture_device(root)
    if recipe.get('device', 'cpu') != execution_device:
        raise ValueError('extractor device differs from captured device')
    device_evidence = None
    if execution_device == 'cuda':
        device_evidence = read_json(features / 'device-runtime.json')
        expected_execution = dict(schema='gotrack_larger_cuda_runtime_v1', device='cuda',
            bootstrap_sha256=hashlib.sha256(_CUDA_BOOTSTRAP.encode()).hexdigest(),
            torch_version='2.5.1+cu124', cuda_version='12.4', deterministic_algorithms=True,
            matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False,
            cudnn_deterministic=True, cublas_workspace_config=':4096:8', threads=1)
        if device_evidence != expected_execution:
            raise ValueError('CUDA extractor actual execution policy differs from capture policy')
    material = dict(schema='gotrack_larger_prelabel_seal_v1', device=execution_device,
        capture_files=[_artifact(root, root / p) for p in ('plan.json', 'capture-index.json', 'capture-receipt.json', 'crops.json')],
        feature_files=[_artifact(features, features / p) for p in ('recipe.json', 'index.json')],
        geometry=[], crops=[], features=[], identities=state_rows(plan), parity=[],
        annotation_inputs=[], test_payload_reads=0)
    if device_evidence is not None:
        material['feature_files'].append(_artifact(features, features / 'device-runtime.json'))
        material['execution_policy'] = device_evidence
    from .quality_adaptation_expanded import _check_parity, validate_retained_geometry
    for captured, extracted in zip(rows, index['rows']):
        if extracted.get('status') not in ('available', 'unavailable'):
            raise ValueError('invalid extraction availability')
        if captured['status'] != 'available':
            if extracted['status'] == 'available':
                raise ValueError('foreign extraction for an unavailable captured state')
            continue
        crop_path = descriptor(root, captured['crops'], verify=True)
        actual_path = descriptor(root, captured['capture'], verify=True)
        crops = load_npz(crop_path, CROP_KEYS)
        validate_crops(crops)
        if crops['query_rgb'].dtype != np.float32 or crops['template_rgb'].dtype != np.float32:
            raise ValueError('refinement states require exact float32 RGB crops')
        if captured['query_rgb_sha256'] != crop_rgb_hash(crops['query_rgb']):
            raise ValueError('capture query RGB identity changed')
        actual = load_npz(actual_path, GEOMETRY_KEYS)
        validate_retained_geometry(actual['crop_from_orig'])
        from .quality_adaptation_prepare import crops_from_retained_capture
        reconstructed = crops_from_retained_capture(dict(query_rgb_crop=crops['query_rgb'].transpose(2, 0, 1)[None],
            template_rgb=crops['template_rgb'].transpose(2, 0, 1)[None], rendered_depth_mm=actual['rendered_depth_mm'],
            rendered_mask=actual['rendered_mask'], crop_k=actual['crop_k'],
            current_crop_camera_from_object_mm=actual['current_crop_camera_from_object_mm']), preserve_float_rgb=True)
        if any(not np.array_equal(reconstructed[k], crops[k]) for k in CROP_KEYS):
            raise ValueError('derived crop geometry differs from retained depth/pose in millimetres')
        if not np.array_equal(crops['query_K'], actual['crop_k']) or not np.array_equal(actual['native_k'], captured['native_calibration']):
            raise ValueError('crop/native calibration differs from retained geometry')
        check = captured.get('sampling_map', {})
        if check.get('integer_centers') != SIZE**2 or not 0 <= check.get('max_native_pixel_difference', -1) < .001:
            raise ValueError('selected iteration lacks a valid sampling map receipt')
        material['crops'].append(dict(id=captured['id'], **captured['crops']))
        material['geometry'].append(dict(id=captured['id'], **captured['capture']))
        if extracted['status'] == 'available':
            _keys(extracted, ('id', 'status', 'crops', 'query_rgb_sha256', 'features'), 'extraction row')
            if extracted['crops'] != captured['crops'] or extracted['query_rgb_sha256'] != captured['query_rgb_sha256']:
                raise ValueError('extractor crop receipt differs from retained input')
            values = load_npz(descriptor(features, extracted['features'], verify=True), FEATURE_KEYS)
            validate_features(values)
            if any(not np.array_equal(values[k], crops[k]) for k in ('template_mask', 'template_xyz_m', 'query_K')):
                raise ValueError('extracted geometry differs from captured crop')
            material['features'].append(dict(id=captured['id'], **extracted['features']))
            material['parity'].append(dict(id=captured['id'], **_check_parity(actual, values, crops['template_mask'])))
    material['available'] = len(material['features'])
    if material['available'] == 0:
        raise ValueError('no extracted crop with frozen prediction parity; cannot seal')
    return material


def seal_features(root, features, *, cloud_execution=False):
    cloud_guard(cloud_execution)
    root, features = Path(root), Path(features)
    remaining(root)
    material = _seal_material(root, features)
    if (root / 'prelabel-seal.json').exists():
        raise FileExistsError('prelabel seal is immutable')
    write_json(root, 'prelabel-seal.json', material)
    return material


def verify_prelabel_seal(root, features):
    seal = read_json(root / 'prelabel-seal.json')
    if seal != _seal_material(root, features):
        raise ValueError('prelabel seal changed: input/capture/crops/geometry/extraction/parity differ')
    return seal




def _save_npz(root, name, arrays):
    resources(root, reserve=16 * 1024**2)
    # Labels share the feature directory, which may exceed the old bridge's
    # two-GiB private limit. The new whole-scratch watchdog owns this bound.
    with atomic_stream(root, root / name, max_file_bytes=16 * 1024**2,
                       max_output_bytes=LIMITS['max_scratch_bytes']) as stream:
        np.savez(stream, **arrays)
    return _artifact(root, root / name)


def _label_plan(path, expected, plan):
    if sha256(path) != require_hash(expected):
        raise ValueError('label plan hash mutation')
    doc = read_json(path)
    _keys(doc, ('schema', 'split_manifest', 'generations'), 'label plan')
    if doc['schema'] != LABEL_SCHEMA or not isinstance(doc['generations'], list):
        raise ValueError('invalid larger label plan')
    expected_sequences = {r['sequence_id'] for r in plan['rows'] if r['input'] is not None}
    result = {}
    # Validate every label identity/path before opening label payloads.
    for row in doc['generations']:
        _keys(row, ('sequence_id', 'generation'), 'generation pin')
        if row['sequence_id'] not in expected_sequences or row['sequence_id'] in result:
            raise ValueError('foreign/duplicate generation identity')
        _pin_shape(row['generation'])
        result[row['sequence_id']] = row['generation']
    if set(result) != expected_sequences:
        raise ValueError('incomplete generation inventory')
    return pinned_file(doc['split_manifest']), {key: pinned_file(pin) for key, pin in result.items()}


def _validate_generation(doc, physical, manifest, split, split_sha, vertices):
    from .quality_ranker_synthetic import NORMALIZED_VIEW_SCHEDULE, validate_normalized_views
    geometry = validate_normalized_views(vertices)
    group = [r for r in split['groups'] if r['sequence_id'] == physical['sequence_id']]
    wanted_split = 'validation' if physical['split'] == 'val' else 'train'
    if (len(group) != 1 or group[0]['split'] != wanted_split
            or group[0]['physical_object_id'] != manifest['physical_object_id']
            or group[0]['mesh_sha256'] != physical['mesh']['sha256']
            or doc.get('kind') != 'quality-ranker-synthetic-generation-v1'
            or doc.get('view_mode') != 'normalized24-v1' or doc.get('view_index') != physical['view_index']
            or doc.get('view_schedule') != [list(r) for r in NORMALIZED_VIEW_SCHEDULE]
            or doc.get('sequence_id') != physical['sequence_id'] or doc.get('split') != wanted_split
            or doc.get('physical_object_id') != manifest['physical_object_id']
            or doc.get('mesh_sha256') != physical['mesh']['sha256']
            or doc.get('video_sha256') != physical['video']['sha256']
            or doc.get('input_sha256') != physical['input']['sha256']
            or doc.get('split_file_sha256') != split_sha
            or doc.get('split_manifest_sha256') != split['manifest_sha256']
            or doc.get('source_sha256') != sha256(ROOT / 'bench/quality_ranker_synthetic.py')
            or not np.array_equal(doc.get('center_m'), geometry['center_m'])
            or doc.get('radius_m') != geometry['radius_m'] or doc.get('distance_m') != geometry['distance_m']
            or doc.get('recipe_sha256') is not None or doc.get('recipe_kind') is not None):
        raise ValueError('normalized generation geometry/identity/split/bytes differ')
    return geometry['poses'][physical['view_index']]


def labels(root, features, label_plan_path, label_plan_sha256, training_source_root,
           training_source_manifest, *, cloud_execution=False):
    cloud_guard(cloud_execution)
    root, features = Path(root), Path(features)
    with stage_lock():
        remaining(root)
        seal = verify_prelabel_seal(root, features)  # Before FIRST label-plan open.
        plan, captures = read_capture(root)
        split_path, generations = _label_plan(Path(label_plan_path), label_plan_sha256, plan)
        shared.verify_training_root(training_source_root, training_source_manifest)
        from .quality_ranker_data import read_split_manifest
        from .quality_adaptation_prepare import project_supervision
        from .quality_adaptation_smoke import render_query_labels
        from .quality_gotrack import TexturedRenderer
        from .quality_runner import read_input
        split = read_split_manifest(split_path)
        extracted = {r['id']: r for r in read_json(features / 'index.json')['rows']}
        inventory = []
        for row in captures:
            item = dict(_base(row), status='unavailable', query_rgb_sha256=row['query_rgb_sha256'],
                failure_reason=row.get('failure_reason') or extracted[row['id']].get('failure_reason') or 'label_not_processed')
            if item['query_rgb_sha256'] is None:
                item['unavailable_provenance'] = row['unavailable_provenance']
            inventory.append(item)
        def persist():
            write_json(features, 'labels.json', dict(schema='correspondence_label_inventory_v1',
                label_origin='synthetic_renderer_training_only', rows=inventory + [plan['sealed_test']]))
        persist()
        try:
            for ordinal, source in enumerate(captures):
                remaining(root)
                if source['status'] != 'available' or extracted[source['id']]['status'] != 'available':
                    continue
                renderer = None
                try:
                    physical = next(r for r in plan['rows'] if source['id'].startswith(r['id'] + ':iteration'))
                    bundle = pinned_file(physical['input']).parent
                    manifest = read_input(bundle)
                    renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'],
                        unlit=True, disable_multisampling=True, coordinate_mode='integer_centers_v1')
                    pose = _validate_generation(read_json(generations[physical['sequence_id']]), physical,
                        manifest, split, sha256(split_path), renderer.vertices_m)
                    crops = load_npz(descriptor(root, source['crops'], verify=True), CROP_KEYS)
                    geometry = load_npz(descriptor(root, source['capture'], verify=True), GEOMETRY_KEYS)
                    query_pose, depth, support = render_query_labels(renderer, object_id=manifest['object_id'],
                        crop_k=crops['query_K'], crop_from_orig=geometry['crop_from_orig'], native_pose=pose)
                    target = project_supervision(template_xyz_m=crops['template_xyz_m'], template_mask=crops['template_mask'],
                        query_K=crops['query_K'], query_T_cam_from_object=query_pose,
                        observed_depth_m=depth, observed_object_mask=support, depth_tolerance_m=.002)
                    inventory[ordinal] = dict(_base(source), status='available', query_rgb_sha256=source['query_rgb_sha256'],
                        labels=_save_npz(features, f'labels-{ordinal:04d}.npz', target))
                except Exception as exc:
                    inventory[ordinal]['failure_reason'] = f'label_failure: {type(exc).__name__}: {exc}'[:1024]
                finally:
                    try:
                        if renderer is not None:
                            renderer.close()
                    finally:
                        persist()
        finally:
            persist()
        shared._run_historical(training_source_root, training_source_manifest, 'bench.quality_adaptation_manifest',
            ['--extraction', features.resolve(), '--labels', (features / 'labels.json').resolve(),
             '--output', (features / 'manifest.json').resolve()], timeout=min(60, remaining(root)))
        ds = load_dataset(features / 'manifest.json')
        write_json(features, 'label-receipt.json', dict(schema='gotrack_larger_labels_receipt_v1',
            prelabel_seal_sha256=sha256(root / 'prelabel-seal.json'), label_plan_sha256=label_plan_sha256,
            label_inventory_sha256=sha256(features / 'labels.json'), manifest_sha256=sha256(features / 'manifest.json'),
            label_artifacts=[dict(id=r['id'], **r['labels']) for r in inventory if r['status'] == 'available'],
            test_payload_reads=0, labels_entered_inference=False, independent_quality_claim=False,
            available=sum(r['status'] == 'available' for r in inventory), parity_rows=len(seal['parity'])))
        return dict(manifest_sha256=sha256(features / 'manifest.json'), rows=len(ds.document['rows']))


def expected_config(device='cpu'):
    from .quality_adaptation_train import TrainConfig
    value = asdict(TrainConfig())
    value['device'] = validate_device(device)
    value.update(epochs=2, max_steps=480, max_seconds=1800., max_total_seconds=3600., patience=2,
        max_output_bytes=1_000_000_000, max_checkpoint_bytes=256 * 1024**2,
        checkpoint_every_steps=40)
    return value


def train_stage(root, features, config, config_sha256, output, training_source_root,
                training_source_manifest, *, cloud_execution=False, resume=False,
                stop_after_steps=None, export_output=None, device='cpu'):
    cloud_guard(cloud_execution)
    root, features, config, output = map(Path, (root, features, config, output))
    _safe_output(output, durable=True)
    resources(root, output, reserve=cfg_reserve() if export_output is None else 96 * 1024**2)
    seconds_left = remaining(root, output)
    verify_prelabel_seal(root, features)
    if sha256(config) != require_hash(config_sha256):
        raise ValueError('training configuration SHA-256 mismatch')
    from .quality_adaptation_train import TrainConfig, read_config
    from dataclasses import asdict
    cfg = read_config(config)
    if asdict(cfg) != expected_config(device):
        raise ValueError('configuration differs from frozen released-init two-epoch recipe')
    if validate_device(device) != capture_device(root):
        raise ValueError('training device differs from authenticated capture device')
    receipt = read_json(features / 'label-receipt.json')
    if (receipt.get('prelabel_seal_sha256') != sha256(root / 'prelabel-seal.json')
            or receipt.get('label_inventory_sha256') != sha256(features / 'labels.json')
            or receipt.get('manifest_sha256') != sha256(features / 'manifest.json')
            or receipt.get('test_payload_reads') != 0 or receipt.get('labels_entered_inference') is not False):
        raise ValueError('label receipt differs from sealed inventory/manifest')
    ds = load_dataset(features / 'manifest.json')
    plan, _ = read_capture(root)
    expected_rows = state_rows(plan)
    if ([r['id'] for r in ds.document['rows']] != [r['id'] for r in expected_rows] + [plan['sealed_test']['id']]
            or ds.document['rows'][-1] != plan['sealed_test']):
        raise ValueError('training requires exact full complete prospective inventory without substitution')
    inventory = read_json(features / 'labels.json')
    extracted = {r['id']: r for r in read_json(features / 'index.json')['rows']}
    if inventory.get('label_origin') != 'synthetic_renderer_training_only' or len(inventory.get('rows', [])) != len(expected_rows) + 1:
        raise ValueError('label inventory schema/origin differs')
    for row, label_row, expected_row in zip(ds.document['rows'][:-1], inventory['rows'][:-1], expected_rows):
        if any(row.get(k) != expected_row[k] for k in ('id', 'object_id', 'split', 'sequence_id', 'capture_group')):
            raise ValueError('training row identity differs from prospective split/frame binding')
        if {k: v for k, v in row.items() if k != 'features'} != label_row:
            raise ValueError('assembled training row differs from authenticated label inventory')
        if row['status'] == 'available' and row['features'] != extracted[row['id']].get('features'):
            raise ValueError('training feature descriptor differs from actual extraction receipt')
    artifacts = [dict(id=r['id'], **r['labels']) for r in inventory['rows'] if r['status'] == 'available']
    if receipt.get('label_artifacts') != artifacts or inventory['rows'][-1] != plan['sealed_test']:
        raise ValueError('label payload descriptors or TEST metadata differ from label receipt')
    if type(resume) is not bool or (stop_after_steps is not None and
            (type(stop_after_steps) is not int or not 1 <= stop_after_steps <= 480)):
        raise ValueError('resume must be boolean; stop-after-steps must be 1..480')
    if resume or export_output is not None:
        run_receipt = read_json(output / 'larger-run-receipt.json')
        if (run_receipt.get('dataset_sha256') != ds.identity
                or run_receipt.get('config_sha256') != canonical_hash(read_json(config))
                or run_receipt.get('label_receipt_sha256') != sha256(features / 'label-receipt.json')):
            raise ValueError('resume/export run differs from authenticated dataset/config/labels')
        for pin in run_receipt['files']:
            path = shared._relative(output, pin['path'])
            if path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
                raise ValueError('resume/export checkpoint or run metadata changed')
    if export_output is not None:
        destination = Path(export_output)
        if destination.resolve().parent != output.resolve() or destination.exists():
            raise ValueError('export must be a new file inside the authenticated run')
        metadata = read_json(output / 'preflight.json')
        if metadata.get('config_sha256') != canonical_hash(read_json(config)) or metadata.get('data', {}).get('dataset_sha256') != ds.identity:
            raise ValueError('export run dataset/config metadata differs')
        command = ['export', '--run', output.resolve(), '--output', destination.resolve(), '--cloud-execution']
    else:
        command = ['resume' if resume else 'train', '--manifest', (features / 'manifest.json').resolve(),
            '--config', config.resolve(), '--output', output.resolve(), '--cloud-execution']
        if stop_after_steps is not None:
            command += ['--stop-after-steps', stop_after_steps]
    result = run_historical_device(training_source_root, training_source_manifest, 'bench.quality_adaptation_train',
        command, timeout=min(cfg.max_seconds, seconds_left),
        accepted_exit_codes=(0,) if export_output is not None else (0, 3), device=device)
    if export_output is None:
        status = read_json(output / 'status.json')
        write_json(output, 'larger-run-receipt.json', dict(
            dataset_sha256=ds.identity, config_sha256=canonical_hash(read_json(config)), device=device,
            label_receipt_sha256=sha256(features / 'label-receipt.json'),
            actual_optimizer_steps=status['steps'], status=status['status'],
            available_train_rows=len(ds.rows('train')), available_used_val_rows=len(ds.rows('val')),
            controlled_quality_claim=False,
            files=[_artifact(output, path) for path in sorted(output.iterdir())
                if path.is_file() and path.name != 'larger-run-receipt.json']))
    remaining(root, output)
    if export_output is not None:
        write_json(output, 'larger-export-device-receipt.json', dict(
            run_device=device, serialization_device='cpu', neural_forward_executed=False,
            historical_exporter_preserved=True, export_sha256=sha256(Path(export_output))))
    return result



def cfg_reserve():
    return 256 * 1024**2  # third atomic checkpoint must fit alongside best and last


def workflow_started(device='cpu'):
    """One immutable wall clock across preparation, capture and training processes."""
    validate_device(device)
    path = Path(SCRATCH) / ('execution-clock.json' if device == 'cpu' else 'execution-clock-cuda-v1.json')
    if device == 'cuda' and not path.exists() and 'VISUALIZEIT_LARGER_CUDA_STARTED' not in os.environ:
        raise ValueError('new CUDA variant requires explicit operator VISUALIZEIT_LARGER_CUDA_STARTED')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(dict(started_unix_s=float(os.environ.get('VISUALIZEIT_LARGER_CUDA_STARTED' if device == 'cuda' else 'VISUALIZEIT_LARGER_STARTED', time.time()))), stream)
        except FileExistsError:
            pass
    started = read_json(path).get('started_unix_s')
    if type(started) not in (int, float) or not np.isfinite(started) or started > time.time() + 1:
        raise ValueError('workflow clock missing/invalid; no reset')
    return started


def supervise(command, *, scratch, durable=None, requested=None, device='cpu'):
    """Bound native stages and descendants; retain full requested-state evidence."""
    import psutil
    scratch = _safe_output(scratch)
    resources(scratch, durable)
    started = workflow_started(device)
    left = LIMITS['max_aggregate_seconds'] - (time.time() - started)
    if left <= 0:
        raise TimeoutError('original aggregate deadline exhausted')
    monitor = Path(SCRATCH) / ('monitor-' + uuid.uuid4().hex)
    monitor.mkdir(parents=True, exist_ok=False)
    write_json(monitor, 'requested-inventory.json', dict(rows=requested or [], test_payload_reads=0))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        VISUALIZEIT_LARGER_STARTED=str(started))
    if device == 'cuda':
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        env['VISUALIZEIT_LARGER_CUDA_STARTED'] = str(started)
    process, reason, peak, descendants = None, None, 0, []
    began = time.time()
    try:
        with (monitor / 'stdout.log').open('xb') as stdout, (monitor / 'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            while True:
                try:
                    root_process = psutil.Process(process.pid)
                    descendants = root_process.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in [root_process, *descendants] if p.is_running())
                    peak = max(peak, rss)
                    if rss > LIMITS['max_process_tree_rss_bytes']:
                        raise MemoryError('eight GiB process-tree RSS ceiling exceeded')
                except psutil.NoSuchProcess:
                    pass
                resources(scratch, durable)
                if time.time() - started > LIMITS['max_aggregate_seconds']:
                    raise TimeoutError('original aggregate deadline exhausted')
                if process.poll() is not None:
                    if process.returncode != 0:
                        raise RuntimeError(f'worker exit {process.returncode}')
                    break
                time.sleep(.25)
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'[:2048]
    finally:
        cleanup_complete = True
        if process is not None:
            try:
                if process.poll() is None:
                    descendants = psutil.Process(process.pid).children(recursive=True)
                    for child in reversed(descendants):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=3)
                for child in descendants:
                    try:
                        if child.is_running():
                            child.kill()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(descendants, timeout=3)
                cleanup_complete = not alive
            except (psutil.Error, subprocess.TimeoutExpired):
                cleanup_complete = False
            lock = ROOT / '.cache/model-quality/.refinement-state-stage.lock'
            if cleanup_complete and lock.is_file() and read_json(lock).get('pid') == process.pid:
                lock.unlink()  # Only this worker's lock after confirmed cleanup.
        if not cleanup_complete:
            reason = 'process_tree_cleanup_incomplete'
        inventory = [dict(row, status='unavailable', failure_reason=reason or 'stage_specific_result_in_worker_receipt') for row in (requested or [])]
        saved = scratch / 'capture-index.json'
        if saved.is_file():
            by_id = {row.get('id'): row for row in read_json(saved).get('rows', [])}
            for row in inventory:
                found = by_id.get(row.get('id'))
                if found is not None and found.get('status') == 'available':
                    row.update(status='available', failure_reason=None, evidence='worker_persisted_capture')
        write_json(monitor, 'state-inventory.json', dict(rows=inventory, test_payload_reads=0))
        write_json(monitor, 'supervision.json', dict(complete=reason is None, failure_reason=reason, command=list(map(str, command)),
            elapsed_seconds=time.time() - began, device=device, execution_variant=device + '-v1', sampled_peak_process_tree_rss_bytes=peak,
            process_tree_cleanup_complete=cleanup_complete, test_payload_reads=0, billing_stopped=None))
    if reason:
        raise RuntimeError(f'bounded stage failed: {reason}; inspect {monitor}')
    return dict(complete=True, monitor=str(monitor))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('validate', 'capture', 'extract', 'seal', 'labels', 'train', 'resume', 'export'):
        p = sub.add_parser(command)
        p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
        p.add_argument('--cloud-execution', action='store_true')
        p.add_argument('--device', choices=('cpu', 'cuda'), default=None)
        p.add_argument('--variant', choices=PREPARATION_VARIANTS, default='legacy-v1')
        if command in ('validate', 'capture'):
            p.add_argument('--plan', type=Path, required=True)
            p.add_argument('--plan-sha256', required=True)
        else:
            p.add_argument('--root', type=Path, required=True)
            p.add_argument('--features', type=Path, required=True)
        if command in ('extract', 'labels', 'train', 'resume', 'export'):
            p.add_argument('--training-source-root', type=Path, required=True)
            p.add_argument('--training-source-manifest', type=Path, required=True)
        if command in ('capture', 'train', 'resume', 'export'):
            p.add_argument('--output', type=Path, required=True)
        if command == 'labels':
            p.add_argument('--label-plan', type=Path, required=True)
            p.add_argument('--label-plan-sha256', required=True)
        if command in ('train', 'resume', 'export'):
            p.add_argument('--config', type=Path, required=True)
            p.add_argument('--config-sha256', required=True)
            if command != 'export':
                p.add_argument('--stop-after-steps', type=int)
            else:
                p.add_argument('--export-output', type=Path, required=True)
    args = parser.parse_args(argv)
    cloud_guard(args.cloud_execution)
    args.device = args.device or (capture_device(args.root) if hasattr(args, 'root') else 'cpu')
    if hasattr(args, 'root') and args.device != capture_device(args.root):
        raise ValueError('selected execution device differs from authenticated capture variant')
    if hasattr(args, 'root'):
        preparation_variant(read_json(args.root / 'capture-receipt.json'), args.variant)
    if args.command == 'validate':
        _, plan = read_plan(args.plan, args.plan_sha256)
        preparation_variant(plan, args.variant)
        preflight(plan)
        result = dict(physical_frames=len(plan['rows']), states=len(state_rows(plan)), test_payload_reads=0)
    elif not args._worker:
        if args.command == 'capture':
            _, plan = read_plan(args.plan, args.plan_sha256)
            preparation_variant(plan, args.variant)
            scratch = args.output
        else:
            plan, _ = read_capture(args.root)
            scratch = args.root
        worker_args = argv if '--device' in argv else [*argv, '--device', args.device]
        result = supervise([sys.executable, '-B', '-m', 'bench.quality_adaptation_larger', *worker_args, '--_worker'],
            scratch=scratch, durable=args.output if args.command in ('train', 'resume', 'export') else None,
            requested=state_rows(plan), device=args.device)
    elif args.command == 'capture':
        result = capture(args.plan, args.plan_sha256, args.output, cloud_execution=True, device=args.device, variant=args.variant)
    elif args.command == 'extract':
        result = extract_stage(args.root, args.features, args.training_source_root, args.training_source_manifest, cloud_execution=True, device=args.device)
    elif args.command == 'labels':
        result = labels(args.root, args.features, args.label_plan, args.label_plan_sha256,
            args.training_source_root, args.training_source_manifest, cloud_execution=True)
    else:
        with stage_lock():
            if args.command == 'seal':
                result = seal_features(args.root, args.features, cloud_execution=True)
            else:
                result = train_stage(args.root, args.features, args.config, args.config_sha256,
                    args.output, args.training_source_root, args.training_source_manifest, cloud_execution=True,
                    resume=args.command == 'resume', stop_after_steps=getattr(args, 'stop_after_steps', None),
                    export_output=getattr(args, 'export_output', None), device=args.device)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    from .quality_adaptation_larger import main as imported_main
    imported_main()
