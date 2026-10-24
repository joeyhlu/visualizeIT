"""Authenticate a saved head and check cached VAL token parity on approved cloud.

Boundary: input-only frozen decoder tokens -> reference DPTHead / installed DPTHead
-> flow and confidence. No RGB network forward, pose replay, labels or accuracy
measurement occurs. Run in a fresh process with an external timeout for kernel hangs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import io
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import time

import numpy as np

from . import quality_adaptation_data as data
from . import quality_adaptation_runtime as runtime
from . import quality_gotrack_inference as installer


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'gotrack_cached_head_artifact_v1'
MANIFEST = ROOT / 'bench/runtime/gotrack-existing-head-20261020.json'
MAX_RECEIPT = 1024**2
FIELDS = frozenset({
    'schema', 'artifact_id', 'export_sha256', 'export_bytes',
    'reference_checkpoint_sha256', 'reference_checkpoint_bytes',
    'training_lock_sha256', 'base_checkpoint_sha256', 'dataset_sha256',
    'training_config_sha256', 'upstream_revision', 'installer_sha256',
    'export_schema', 'export_deployment', 'head_interface', 'quality_claim',
})


def read_manifest(path: Path, expected_sha256: str) -> tuple[dict, str]:
    """Parse only the exact bounded bytes authenticated by the operator."""
    data.require_hash(expected_sha256, 'manifest SHA256')
    path = installer._checked_regular_file(path, 'artifact manifest')
    raw = installer._read_limited(path, data.MAX_JSON, 'artifact manifest')
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('artifact manifest digest mismatch')
    value = installer._loads_unique(raw, 'artifact manifest')
    if not isinstance(value, dict) or set(value) != FIELDS or value['schema'] != SCHEMA:
        raise ValueError('artifact manifest schema or fields differ')
    for key in FIELDS:
        if key.endswith('_sha256'):
            data.require_hash(value[key], key)
    if (not isinstance(value['artifact_id'], str) or not 1 <= len(value['artifact_id']) <= 128
            or not isinstance(value['upstream_revision'], str)
            or len(value['upstream_revision']) != 40
            or any(c not in '0123456789abcdef' for c in value['upstream_revision'])
            or value['export_schema'] != installer.EXPORT_SCHEMA
            or value['export_deployment'] != installer.EXPORT_DEPLOYMENT
            or value['head_interface'] != installer._LEGACY_TRAINING['head_interface']
            or value['quality_claim'] is not False):
        raise ValueError('artifact manifest interface/provenance differs')
    for key, limit in [('export_bytes', 128 * 1024**2), ('reference_checkpoint_bytes', 384 * 1024**2)]:
        if type(value[key]) is not int or not 0 < value[key] <= limit:
            raise ValueError('artifact byte count outside bound')
    return value, expected_sha256


@dataclass(frozen=True)
class HeadContext:
    """Installer context, with no admission to the historical geometry profiles."""
    head_path: Path
    requested_head_sha256: str
    artifact: MappingProxyType
    manifest_sha256: str

    def identity_dict(self, *, installed=False):
        return {'artifact_id': self.artifact['artifact_id'],
                'artifact_manifest_sha256': self.manifest_sha256,
                'head': {'requested_sha256': self.requested_head_sha256,
                         'installed_sha256': self.requested_head_sha256 if installed else None},
                'quality_claim': False, 'geometry_admission': False}


def context_for_export(record: dict, manifest_sha256: str, export: Path) -> HeadContext:
    """Use the production atomic installer without extending its old registry."""
    loaded = installer._checked_regular_file(installer.__file__, 'loaded atomic installer')
    if data.sha256(loaded) != record['installer_sha256']:
        raise ValueError('loaded atomic installer source mismatch')
    training = MappingProxyType({k: record[k] for k in (
        'training_lock_sha256', 'base_checkpoint_sha256', 'dataset_sha256',
        'training_config_sha256', 'export_deployment')})
    artifact = MappingProxyType({'artifact_id': record['artifact_id'],
        'export_sha256': record['export_sha256'], 'export_bytes': record['export_bytes'],
        'training': training})
    return HeadContext(Path(export), record['export_sha256'], artifact, manifest_sha256)


def authenticated_bytes(path: Path, size: int, digest: str, name: str) -> bytes:
    path = installer._checked_regular_file(path, name)
    raw = installer._read_limited(path, size, name)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f'{name} digest or size mismatch')
    return raw


class Budget:
    def __init__(self, deadline_utc: str, max_seconds: float):
        if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
                or not math.isfinite(max_seconds) or not 0 < max_seconds <= 900):
            raise ValueError('max_seconds must be finite and in (0,900]')
        if not isinstance(deadline_utc, str) or not deadline_utc.endswith('Z'):
            raise ValueError('deadline must be an explicit UTC timestamp ending Z')
        self.deadline = datetime.fromisoformat(deadline_utc[:-1] + '+00:00')
        self.started = time.monotonic()
        self.max_seconds = max_seconds
        self.check()

    def check(self):
        if (datetime.now(timezone.utc) >= self.deadline
                or time.monotonic() - self.started >= self.max_seconds):
            raise TimeoutError('cached head parity deadline or invocation budget expired')


def select_validation(ds, maximum: int) -> tuple[list, dict]:
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError('max_rows must be an integer in 1..8')
    all_rows = ds.rows('val', available=False)
    available = sorted((r for r in all_rows if r['status'] == 'available'), key=lambda r: r['id'])
    if not available:
        raise ValueError('no available VAL token rows')
    return available[:maximum], {'total': len(all_rows), 'available': len(available),
        'unavailable': len(all_rows) - len(available), 'selected': min(maximum, len(available)),
        'selection': 'first_available_VAL_ids_sorted_lexically'}


def feature_tokens(ds, row, device, torch):
    if row['split'] != 'val' or row['status'] != 'available':
        raise ValueError('only available VAL features may be opened')
    values = data.load_npz(data.descriptor(ds.root, row['features'], verify=True), data.FEATURE_KEYS)
    data.validate_features(values)
    tokens = [None] * 12
    for hook in data.HOOKS:
        tokens[hook] = torch.from_numpy(values[f'tokens_{hook}'][None]).to(device)
    return tokens


def checked_outputs(outputs) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise ValueError('head must return exactly flow and confidence')
    arrays = tuple(v.detach().cpu().numpy().copy() for v in outputs)
    for value, shape in zip(arrays, ((1, 2, data.SIZE, data.SIZE), (1, data.SIZE, data.SIZE))):
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('head output shape/dtype/nonfinite mismatch')
    if np.any((arrays[1] < 0) | (arrays[1] > 1)):
        raise ValueError('head confidence outside [0,1]')
    return arrays


def compare_outputs(reference, actual) -> dict:
    errors = [float(np.max(np.abs(a.astype(np.float64) - b))) for a, b in zip(reference, actual)]
    if any(error > 1e-6 for error in errors):
        raise ValueError(f'cached token parity failed: max absolute differences {errors}')
    return {'flow_max_absolute_difference_px': errors[0], 'confidence_max_absolute_difference': errors[1]}


def _sync(torch, device):
    if device == 'cuda':
        torch.cuda.synchronize()


def _release(torch, device):
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


def execute(record, context, dataset: Path, reference_checkpoint: Path, *, device, max_rows, budget, receipt):
    """Actual execution path; mocks are confined to the separate test module."""
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    budget.check()
    versions = runtime.runtime_versions()
    source = runtime.verify_sources()
    if (runtime.ROOT.resolve() != ROOT.resolve()
            or source['source_lock_sha256'] != record['training_lock_sha256']
            or source['upstream_revision'] != record['upstream_revision']
            or runtime.GOTRACK_CHECKPOINT != record['base_checkpoint_sha256']):
        raise ValueError('runtime root/source identity differs from artifact')
    # Existing receipt validation reads recipe/index metadata, never annotations.
    from .quality_adaptation_train import validate_feature_receipt
    ds = data.load_dataset(dataset, inspect_arrays=False)
    if ds.identity != record['dataset_sha256']:
        raise ValueError('dataset identity differs from artifact')
    validate_feature_receipt(ds)
    rows, availability = select_validation(ds, max_rows)
    receipt.update(versions=versions, source=source, dataset_sha256=ds.identity,
                   validation_inventory=availability, selected_row_ids=[r['id'] for r in rows],
                   selected_features=[{'id': r['id'], 'features': dict(r['features'])} for r in rows])
    budget.check()
    # Fail on export corruption before reference model construction.
    del_export = authenticated_bytes(context.head_path, record['export_bytes'], record['export_sha256'], 'head export')
    del del_export
    raw = authenticated_bytes(reference_checkpoint, record['reference_checkpoint_bytes'],
                              record['reference_checkpoint_sha256'], 'reference checkpoint')
    budget.check()
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested but unavailable')
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    del raw
    runtime.check_resume(state, identity=ds.identity, config_hash=record['training_config_sha256'],
                         source_hash=record['training_lock_sha256'])
    if state.get('versions') != versions:
        raise ValueError('reference checkpoint runtime versions differ')
    budget.check()
    head = runtime.load_head(initialize=False, device='cpu')
    reference = []
    timings = {'reference_forward_seconds': [], 'installed_forward_seconds': []}
    receipt['timings'] = timings
    try:
        installer._validate_state_dict(state.get('head'), head.state_dict(), torch)
        head.load_state_dict(state['head'], strict=True)
        del state
        head.eval().requires_grad_(False)
        head.to(device)
        with torch.inference_mode():
            for row in rows:
                receipt['active_row_id'] = row['id']
                budget.check()
                tokens = feature_tokens(ds, row, device, torch)
                budget.check()
                _sync(torch, device)
                started = time.monotonic()
                outputs = head(tokens, (data.SIZE, data.SIZE))
                _sync(torch, device)
                timings['reference_forward_seconds'].append(time.monotonic() - started)
                reference.append(checked_outputs(outputs))
                del tokens, outputs
                budget.check()
    finally:
        del head
        _release(torch, device)
    budget.check()
    network = SimpleNamespace(pose_head=runtime.load_head(initialize=False, device='cpu'))
    try:
        budget.check()
        network, provenance = installer.install_adapted_head(network, context, device=device, torch_module=torch)
        receipt['installation'] = provenance
        receipt['comparisons'] = []
        with torch.inference_mode():
            for row, expected in zip(rows, reference):
                receipt['active_row_id'] = row['id']
                budget.check()
                tokens = feature_tokens(ds, row, device, torch)
                budget.check()
                _sync(torch, device)
                started = time.monotonic()
                outputs = network.pose_head(tokens, (data.SIZE, data.SIZE))
                _sync(torch, device)
                timings['installed_forward_seconds'].append(time.monotonic() - started)
                actual = checked_outputs(outputs)
                receipt['comparisons'].append({'id': row['id'], **compare_outputs(expected, actual)})
                del tokens, outputs, actual
                budget.check()
        receipt['cached_token_parity_passed'] = True
        receipt['memory'] = {'cuda_current_allocated_bytes': int(torch.cuda.memory_allocated()) if device == 'cuda' else None,
                             'peak_memory_bytes': None, 'peak_memory_measured': False}
    finally:
        del network
        _release(torch, device)


def run(*, artifact_manifest, artifact_manifest_sha256, export, reference_checkpoint,
        dataset, output, deadline_utc, max_seconds, device='cpu', max_rows=3, cloud_execution=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {'schema': 'gotrack_cached_head_parity_receipt_v1', 'status': 'failed',
        'cached_token_parity_passed': False, 'full_network_forward_executed': False,
        'pose_replay': False, 'quality_claim': False, 'test_payload_reads': 0,
        'label_payload_reads': 0, 'device': device, 'deadline_utc': deadline_utc,
        'max_seconds': max_seconds if isinstance(max_seconds, (int, float)) and math.isfinite(max_seconds) else str(max_seconds),
        'max_rows': max_rows,
        'evidence_scope': 'actual_DPT_cached_VAL_tokens_only', 'mocked_execution': False}
    try:
        runtime.cloud_guard(cloud_execution)  # Before any Torch import or model construction.
        budget = Budget(deadline_utc, max_seconds)
        record, manifest_hash = read_manifest(Path(artifact_manifest), artifact_manifest_sha256)
        receipt['artifact'] = record
        receipt['artifact_manifest_sha256'] = manifest_hash
        context = context_for_export(record, manifest_hash, Path(export))
        execute(record, context, Path(dataset), Path(reference_checkpoint), device=device,
                max_rows=max_rows, budget=budget, receipt=receipt)
        receipt['status'] = 'passed'
    except Exception as exc:
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)[:2048]}
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        runtime.write_json(output, 'receipt.json', receipt, max_output_bytes=MAX_RECEIPT)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-manifest', type=Path, default=MANIFEST)
    parser.add_argument('--artifact-manifest-sha256', required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--max-rows', type=int, default=3)
    parser.add_argument('--cloud-execution', action='store_true')
    args = vars(parser.parse_args(argv))
    result = run(**args)
    print(result['status'])
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
