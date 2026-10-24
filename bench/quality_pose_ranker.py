"""Deterministic and compact learned pose-candidate rankers.

Rankers receive only candidates that passed the tracker’s current-image,
backend-motion, and causal-motion gates. Checkpoints use NumPy for inference;
loading or calling this module does not import PyTorch.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
import json
import math
import re

import numpy as np

from .quality_ranker_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_VECTOR_SHAPE,
)


_CHECKPOINT_SCHEMA_VERSION = 1
_INTEGRATION_CHECKPOINT_SCHEMA_VERSION = 2
_CONFIDENCE_CHECKPOINT_SCHEMA_VERSION = 3
_INTEGRATION_ARTIFACT_PURPOSE = "integration_smoke_v1"
_CONFIDENCE_ARTIFACT_PURPOSE = "confidence_rejection_research_v1"
_CONFIDENCE_POLICY = "legacy_veto_v1"
_INTEGRATION_CALIBRATION_UNAVAILABLE_REASON = "integration_smoke_does_not_fit_calibration"
_MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
_V1_CHECKPOINT_FIELDS = frozenset((
    'schema_version', 'feature_schema_version', 'feature_names',
    'mean', 'scale', 'layers', 'architecture', 'acceptance_probability',
    'distinct_pose_margin_probability', 'provenance',
))
_V2_CHECKPOINT_FIELDS = frozenset((
    'schema_version', 'artifact_purpose', 'selection_enabled',
    'calibration_unavailable_reason', 'feature_schema_version',
    'feature_names', 'mean', 'scale', 'layers', 'architecture', 'provenance',
))
_V3_CHECKPOINT_FIELDS = frozenset((
    'schema_version', 'artifact_purpose', 'policy', 'research_only',
    'feature_schema_version', 'feature_names', 'architecture', 'mean', 'scale',
    'layers', 'acceptance_probability', 'provenance',
))
_V4_CHECKPOINT_FIELDS = _V1_CHECKPOINT_FIELDS | frozenset((
    'artifact_purpose', 'policy', 'research_only',
))
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_TRAINER_SOURCE_FILES = frozenset((
    'quality_ranker_train.py', 'quality_ranker_features.py',
    'quality_pose_ranker.py', 'quality_ranker_data.py',
    'quality_contract.py', 'quality_gotrack.py',
))
_RESEARCH_SOURCE_FILES = _TRAINER_SOURCE_FILES | frozenset((
    'quality_confidence_rejection.py', 'quality_confidence_evaluate.py',
    'quality_pipeline.py', 'quality_runner.py', 'quality_offline.py',
))
_DATA_IDENTITY_HASH_FIELDS = frozenset((
    'feature_cache_sha256', 'label_cache_sha256', 'split_sha256',
    'source_trace_sha256', 'reference_labels_sha256',
    'cache_manifest_sha256',
))


def _is_sha256(value):
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _validate_calibrated_provenance(document):
    """Require the exact validation-only provenance emitted by the trainer."""
    provenance = document.get('provenance')
    required = {'identity', 'checkpoint_sha256', 'calibration',
                'calibration_split', 'test_read'}
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError('learned checkpoint needs the frozen trainer provenance fields')
    if provenance['calibration_split'] != 'validation':
        raise ValueError('learned checkpoint calibration must use validation data')
    if provenance['test_read'] is not False:
        raise ValueError('learned checkpoint must certify that test data was not read')
    if not _is_sha256(provenance['checkpoint_sha256']):
        raise ValueError('trainer checkpoint_sha256 must be a lowercase SHA-256 digest')

    identity = provenance['identity']
    identity_fields = {
        'manifest_sha256', 'data', 'source', 'config_sha256',
        'feature_names', 'feature_schema_version', 'runtime',
    }
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise ValueError('trainer identity fields do not match the frozen schema')
    if (not _is_sha256(identity['manifest_sha256']) or
            not _is_sha256(identity['config_sha256'])):
        raise ValueError('trainer identity hashes must be lowercase SHA-256 digests')
    if (type(identity['feature_schema_version']) is not int or
            identity['feature_schema_version'] != FEATURE_SCHEMA_VERSION or
            identity['feature_names'] != list(FEATURE_NAMES)):
        raise ValueError('trainer identity feature schema does not match this ranker')

    data = identity['data']
    data_fields = _DATA_IDENTITY_HASH_FIELDS | {
        'feature_schema_version', 'feature_names'}
    if not isinstance(data, dict) or set(data) != data_fields:
        raise ValueError('trainer data identity fields do not match the frozen schema')
    if any(not _is_sha256(data[name]) for name in _DATA_IDENTITY_HASH_FIELDS):
        raise ValueError('trainer data identity hashes must be lowercase SHA-256 digests')
    if (type(data['feature_schema_version']) is not int or
            data['feature_schema_version'] != FEATURE_SCHEMA_VERSION or
            data['feature_names'] != list(FEATURE_NAMES)):
        raise ValueError('trainer data feature schema does not match this ranker')

    source = identity['source']
    if not isinstance(source, dict) or set(source) != _TRAINER_SOURCE_FILES:
        raise ValueError('trainer source identity does not match the frozen file set')
    if any(not _is_sha256(value) for value in source.values()):
        raise ValueError('trainer source hashes must be lowercase SHA-256 digests')

    runtime = identity['runtime']
    runtime_fields = {'torch', 'numpy', 'python', 'cuda', 'cudnn', 'device'}
    if not isinstance(runtime, dict) or set(runtime) != runtime_fields:
        raise ValueError('trainer runtime identity fields do not match the frozen schema')
    if any(not isinstance(runtime[name], str) or not runtime[name]
           for name in ('torch', 'numpy', 'python', 'device')):
        raise ValueError('trainer runtime version and device fields must be nonempty text')
    if runtime['cuda'] is not None and not isinstance(runtime['cuda'], str):
        raise ValueError('trainer CUDA version must be text or null')
    if runtime['cudnn'] is not None and type(runtime['cudnn']) is not int:
        raise ValueError('trainer cuDNN version must be an integer or null')

    calibration = provenance['calibration']
    calibration_fields = {
        'temperature', 'bias', 'acceptance_probability', 'validation_bce',
        'validation_balanced_accuracy',
    }
    if not isinstance(calibration, dict) or set(calibration) != calibration_fields:
        raise ValueError('validation calibration fields do not match the frozen schema')
    if any(type(value) not in (int, float) or not math.isfinite(float(value))
           for value in calibration.values()):
        raise ValueError('validation calibration values must be finite numbers')
    if float(calibration['temperature']) <= 0.:
        raise ValueError('validation calibration temperature must be positive')
    if float(calibration['validation_bce']) < 0.:
        raise ValueError('validation calibration BCE must be nonnegative')
    if not 0. <= float(calibration['validation_balanced_accuracy']) <= 1.:
        raise ValueError('validation balanced accuracy must be between zero and one')
    acceptance = float(document['acceptance_probability'])
    if float(calibration['acceptance_probability']) != acceptance:
        raise ValueError('calibrated and checkpoint acceptance probabilities disagree')


def _validate_integration_provenance(document):
    """Validate the strict non-selecting score-only schema-v2 artifact."""
    if (type(document.get('schema_version')) is not int or
            document['schema_version'] != _INTEGRATION_CHECKPOINT_SCHEMA_VERSION):
        raise ValueError('unsupported integration ranker checkpoint schema_version')
    if document.get('artifact_purpose') != _INTEGRATION_ARTIFACT_PURPOSE:
        raise ValueError('integration checkpoint artifact purpose is missing or contradictory')
    if document.get('selection_enabled') is not False:
        raise ValueError('integration checkpoint selection_enabled must be false')
    if document.get('calibration_unavailable_reason') != _INTEGRATION_CALIBRATION_UNAVAILABLE_REASON:
        raise ValueError('integration checkpoint calibration-unavailable reason is missing or contradictory')
    provenance = document.get('provenance')
    required_provenance = {'artifact_purpose', 'identity', 'checkpoint_sha256', 'test_read'}
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ValueError('integration checkpoint provenance fields do not match schema v2')
    if provenance['artifact_purpose'] != _INTEGRATION_ARTIFACT_PURPOSE:
        raise ValueError('integration provenance purpose is missing or contradictory')
    if provenance['test_read'] is not False:
        raise ValueError('integration checkpoint must certify that test data was not read')
    if not _is_sha256(provenance['checkpoint_sha256']):
        raise ValueError('integration checkpoint_sha256 must be a lowercase SHA-256 digest')

    identity = provenance['identity']
    identity_fields = {
        'manifest_sha256', 'data', 'source', 'config_sha256', 'purpose',
        'feature_names', 'feature_schema_version', 'runtime',
    }
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise ValueError('integration trainer identity fields do not match schema v2')
    if identity['purpose'] != _INTEGRATION_ARTIFACT_PURPOSE:
        raise ValueError('integration trainer identity purpose is missing or contradictory')
    if (not _is_sha256(identity['manifest_sha256']) or
            not _is_sha256(identity['config_sha256'])):
        raise ValueError('integration trainer identity hashes must be lowercase SHA-256 digests')
    if (type(identity['feature_schema_version']) is not int or
            identity['feature_schema_version'] != FEATURE_SCHEMA_VERSION or
            identity['feature_names'] != list(FEATURE_NAMES)):
        raise ValueError('integration trainer feature schema does not match this ranker')

    data = identity['data']
    data_fields = _DATA_IDENTITY_HASH_FIELDS | {'feature_schema_version', 'feature_names'}
    if not isinstance(data, dict) or set(data) != data_fields:
        raise ValueError('integration trainer data identity fields do not match schema v2')
    if any(not _is_sha256(data[name]) for name in _DATA_IDENTITY_HASH_FIELDS):
        raise ValueError('integration trainer data identity hashes must be lowercase SHA-256 digests')
    if (type(data['feature_schema_version']) is not int or
            data['feature_schema_version'] != FEATURE_SCHEMA_VERSION or
            data['feature_names'] != list(FEATURE_NAMES)):
        raise ValueError('integration trainer data feature schema does not match this ranker')

    source = identity['source']
    if not isinstance(source, dict) or set(source) != _TRAINER_SOURCE_FILES:
        raise ValueError('integration trainer source identity does not match the frozen file set')
    if any(not _is_sha256(value) for value in source.values()):
        raise ValueError('integration trainer source hashes must be lowercase SHA-256 digests')
    runtime = identity['runtime']
    runtime_fields = {'torch', 'numpy', 'python', 'cuda', 'cudnn', 'device'}
    if not isinstance(runtime, dict) or set(runtime) != runtime_fields:
        raise ValueError('integration trainer runtime identity does not match schema v2')
    if any(not isinstance(runtime[name], str) or not runtime[name]
           for name in ('torch', 'numpy', 'python', 'device')):
        raise ValueError('integration trainer runtime fields must be nonempty text')
    if runtime['cuda'] is not None and not isinstance(runtime['cuda'], str):
        raise ValueError('integration trainer CUDA version must be text or null')
    if runtime['cudnn'] is not None and type(runtime['cudnn']) is not int:
        raise ValueError('integration trainer cuDNN version must be an integer or null')
    if runtime['device'] != 'cpu' or not runtime['torch'].startswith('2.5.1'):
        raise ValueError('integration artifacts require the pinned PyTorch 2.5.1 CPU execution path')
    if document.get('architecture') != 'linear':
        raise ValueError('integration artifacts must use the approved linear architecture')


def _validate_confidence_provenance(document):
    """Validate the strict research-only legacy-veto schema-v3 artifact."""
    if type(document.get('schema_version')) is not int or document['schema_version'] != 3:
        raise ValueError('unsupported confidence research checkpoint schema_version')
    if document.get('artifact_purpose') != _CONFIDENCE_ARTIFACT_PURPOSE:
        raise ValueError('confidence research artifact purpose is missing or contradictory')
    if document.get('policy') != _CONFIDENCE_POLICY:
        raise ValueError('confidence research policy is missing or contradictory')
    if document.get('research_only') is not True:
        raise ValueError('confidence research artifact must be marked research_only')
    provenance = document.get('provenance')
    required = {
        'identity', 'checkpoint_sha256', 'label_policy', 'supervision_counts',
        'calibration', 'calibration_split', 'calibration_scope',
        'calibration_rule', 'checkpoint_selection', 'development_fitting',
        'test_read', 'unsupported_claims',
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError('confidence research provenance fields do not match schema v3')
    if not _is_sha256(provenance['checkpoint_sha256']):
        raise ValueError('confidence checkpoint_sha256 must be a lowercase SHA-256 digest')
    if provenance['calibration_split'] != 'validation':
        raise ValueError('confidence calibration must use validation data')
    if provenance['calibration_scope'] != 'legacy_selected_labeled_eligible':
        raise ValueError('confidence calibration scope is contradictory')
    if provenance['calibration_rule'] != 'fixed_grid_bce_then_balanced_accuracy_conservative_v1':
        raise ValueError('confidence calibration rule is contradictory')
    if provenance['checkpoint_selection'] != 'legacy_selected_validation_uncalibrated_bce':
        raise ValueError('confidence checkpoint-selection rule is contradictory')
    if provenance['development_fitting'] is not True or provenance['test_read'] is not False:
        raise ValueError('confidence artifact must disclose development fitting and no test read')
    expected_claims = [
        'candidate_ordering', 'independent_test_accuracy', 'object_generalization',
        'physical_attachment_accuracy',
    ]
    if provenance['unsupported_claims'] != expected_claims:
        raise ValueError('confidence unsupported_claims differ from the research contract')

    identity = provenance['identity']
    identity_fields = {
        'manifest_sha256', 'data', 'source', 'config', 'config_sha256', 'purpose', 'policy',
        'label_policy_sha256', 'feature_names', 'feature_schema_version', 'runtime',
    }
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise ValueError('confidence trainer identity fields do not match schema v3')
    if identity['purpose'] != _CONFIDENCE_ARTIFACT_PURPOSE or identity['policy'] != _CONFIDENCE_POLICY:
        raise ValueError('confidence trainer purpose/policy is contradictory')
    if any(not _is_sha256(identity[name]) for name in
           ('manifest_sha256', 'config_sha256', 'label_policy_sha256')):
        raise ValueError('confidence trainer identity hashes must be lowercase SHA-256 digests')
    config = identity['config']
    config_fields = {
        'architecture', 'seed', 'epochs', 'batch_size', 'accumulation',
        'learning_rate', 'weight_decay', 'pairwise_weight', 'patience',
        'min_delta', 'max_seconds', 'max_output_bytes', 'max_checkpoint_bytes',
        'max_candidates_per_group', 'max_feature_rows', 'amp', 'device',
    }
    if not isinstance(config, dict) or set(config) != config_fields:
        raise ValueError('confidence trainer config fields do not match schema v3')
    expected_config = {
        'architecture': document.get('architecture'), 'seed': 42,
        'epochs': config.get('epochs'), 'batch_size': 8, 'accumulation': 4,
        'learning_rate': .001, 'weight_decay': .0001, 'pairwise_weight': .5,
        'patience': 6, 'min_delta': .00001, 'max_seconds': 300.,
        'max_output_bytes': 64 * 1024 * 1024,
        'max_checkpoint_bytes': 16 * 1024 * 1024,
        'max_candidates_per_group': 256, 'max_feature_rows': 100000,
        'amp': False, 'device': 'cpu',
    }
    if type(config.get('epochs')) is not int or config['epochs'] not in (2, 40):
        raise ValueError('confidence trainer epochs must match research or smoke config')
    if config != expected_config:
        raise ValueError('confidence trainer config differs from the frozen CPU contract')
    config_hash = sha256(json.dumps(
        {'purpose': _CONFIDENCE_ARTIFACT_PURPOSE, 'config': config},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if config_hash != identity['config_sha256']:
        raise ValueError('confidence trainer config hash differs from declared config')
    if (type(identity['feature_schema_version']) is not int or
            identity['feature_schema_version'] != FEATURE_SCHEMA_VERSION or
            identity['feature_names'] != list(FEATURE_NAMES)):
        raise ValueError('confidence trainer feature schema does not match this ranker')
    data = identity['data']
    data_fields = _DATA_IDENTITY_HASH_FIELDS | {'feature_schema_version', 'feature_names'}
    if not isinstance(data, dict) or set(data) != data_fields:
        raise ValueError('confidence data identity fields do not match schema v3')
    if any(not _is_sha256(data[name]) for name in _DATA_IDENTITY_HASH_FIELDS):
        raise ValueError('confidence data identity hashes must be lowercase SHA-256 digests')
    if (type(data['feature_schema_version']) is not int or
            data['feature_schema_version'] != FEATURE_SCHEMA_VERSION or
            data['feature_names'] != list(FEATURE_NAMES)):
        raise ValueError('confidence data feature schema does not match this ranker')
    source = identity['source']
    if not isinstance(source, dict) or set(source) != _RESEARCH_SOURCE_FILES:
        raise ValueError('confidence source identity does not match the research file set')
    if any(not _is_sha256(value) for value in source.values()):
        raise ValueError('confidence source hashes must be lowercase SHA-256 digests')
    for name, expected in source.items():
        path = Path(__file__).with_name(name)
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'confidence research source hash mismatch: {name}')
    runtime = identity['runtime']
    runtime_fields = {'torch', 'numpy', 'python', 'cuda', 'cudnn', 'device'}
    if not isinstance(runtime, dict) or set(runtime) != runtime_fields:
        raise ValueError('confidence runtime identity fields do not match schema v3')
    if any(not isinstance(runtime[name], str) or not runtime[name]
           for name in ('torch', 'numpy', 'python', 'device')):
        raise ValueError('confidence runtime fields must be nonempty text')
    if runtime['cuda'] is not None and not isinstance(runtime['cuda'], str):
        raise ValueError('confidence CUDA version must be text or null')
    if runtime['cudnn'] is not None and type(runtime['cudnn']) is not int:
        raise ValueError('confidence cuDNN version must be an integer or null')
    if (re.fullmatch(r'2\.5\.1(?:\+cu124)?', runtime['torch']) is None or
            runtime['numpy'] != '1.26.4' or runtime['device'] != 'cpu'):
        raise ValueError('confidence runtime must be pinned PyTorch 2.5.1(+cu124), NumPy 1.26.4 CPU')

    label_policy = provenance['label_policy']
    label_fields = {'sha256', 'frozen', 'quality_target_definition',
                    'angular_observability_interpretation'}
    if not isinstance(label_policy, dict) or set(label_policy) != label_fields:
        raise ValueError('confidence label-policy provenance is incomplete')
    if label_policy['sha256'] != identity['label_policy_sha256'] or not _is_sha256(label_policy['sha256']):
        raise ValueError('confidence frozen label-policy hash differs from identity')
    if not isinstance(label_policy['frozen'], dict):
        raise ValueError('confidence frozen label policy must be an object')
    try:
        frozen_policy_hash = sha256(json.dumps(
            label_policy['frozen'], sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode('utf-8')).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError('confidence frozen label policy must be finite JSON') from exc
    if frozen_policy_hash != label_policy['sha256']:
        raise ValueError('confidence frozen label-policy content hash differs')
    if label_policy['quality_target_definition'] != 'frozen_available_dimension_quality':
        raise ValueError('confidence quality target definition is contradictory')
    if label_policy['angular_observability_interpretation'] != 'ambiguous_frozen_before_inference_rotation_excluded_unchanged':
        raise ValueError('confidence angular observability interpretation is contradictory')
    counts = provenance['supervision_counts']
    if not isinstance(counts, dict) or set(counts) != {'train', 'validation'}:
        raise ValueError('confidence supervision counts are incomplete')
    for split in ('train', 'validation'):
        row = counts[split]
        fields = {'eligible_labeled', 'eligible_good', 'eligible_bad',
                  'legacy_selected_labeled', 'legacy_selected_good', 'legacy_selected_bad'}
        if (not isinstance(row, dict) or set(row) != fields or
                any(type(value) is not int or value < 0 for value in row.values())):
            raise ValueError('confidence supervision counts are malformed')
        if (row['eligible_good'] + row['eligible_bad'] != row['eligible_labeled'] or
                row['legacy_selected_good'] + row['legacy_selected_bad'] !=
                row['legacy_selected_labeled'] or
                row['legacy_selected_labeled'] > row['eligible_labeled'] or
                row['legacy_selected_good'] > row['eligible_good'] or
                row['legacy_selected_bad'] > row['eligible_bad']):
            raise ValueError('confidence supervision counts are incoherent')
        required = (('eligible_good', 'eligible_bad',
                     'legacy_selected_good', 'legacy_selected_bad')
                    if split == 'train' else
                    ('legacy_selected_good', 'legacy_selected_bad'))
        if any(row[name] <= 0 for name in required):
            raise ValueError('confidence supervision must retain both research classes')
    calibration = provenance['calibration']
    calibration_fields = {'temperature', 'bias', 'acceptance_probability',
                          'validation_bce', 'validation_balanced_accuracy'}
    if not isinstance(calibration, dict) or set(calibration) != calibration_fields:
        raise ValueError('confidence calibration fields do not match the research schema')
    if any(type(value) not in (int, float) or isinstance(value, bool) or
           not math.isfinite(float(value)) for value in calibration.values()):
        raise ValueError('confidence calibration values must be finite numbers')
    if float(calibration['temperature']) <= 0. or float(calibration['validation_bce']) < 0.:
        raise ValueError('confidence calibration temperature/BCE is invalid')
    if not 0. <= float(calibration['validation_balanced_accuracy']) <= 1.:
        raise ValueError('confidence validation balanced accuracy is invalid')
    if float(calibration['acceptance_probability']) != float(document['acceptance_probability']):
        raise ValueError('confidence calibration and artifact threshold disagree')


def _immutable_array(value, shape, name):
    try:
        original = np.asarray(value, dtype=object)
    except Exception as exc:
        raise ValueError(f'{name} must be a numeric array') from exc
    if original.shape != shape or any(
            isinstance(item, (bool, np.bool_)) or
            not isinstance(item, (int, float, np.integer, np.floating))
            for item in original.flat):
        raise ValueError(f'{name} must have shape {shape} and contain real numbers')
    array = np.ascontiguousarray(original, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite values')
    backing = array.tobytes(order='C')
    return np.frombuffer(backing, dtype=np.float64).reshape(shape)


def _freeze_json(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('provenance must be finite JSON-compatible data')


def _thaw_json(value):
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _snapshot_checkpoint(checkpoint):
    if isinstance(checkpoint, Mapping):
        try:
            encoded = json.dumps(dict(checkpoint), allow_nan=False,
                                 separators=(',', ':')).encode('utf-8')
            if len(encoded) > _MAX_CHECKPOINT_BYTES:
                raise ValueError('checkpoint exceeds the 16 MiB size limit')
            document = json.loads(encoded)
        except ValueError as exc:
            if '16 MiB size limit' in str(exc):
                raise
            raise ValueError('checkpoint mapping must be finite JSON data') from exc
        except TypeError as exc:
            raise ValueError('checkpoint mapping must be finite JSON data') from exc
    else:
        path = Path(checkpoint)
        try:
            size = path.stat().st_size
            if size < 0 or size > _MAX_CHECKPOINT_BYTES:
                raise ValueError('checkpoint exceeds the 16 MiB size limit')
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f'cannot read ranker checkpoint: {path.name}') from exc
        if len(payload) != size:
            raise ValueError('checkpoint changed while being read')
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError('checkpoint must be valid UTF-8 JSON') from exc
    if not isinstance(document, dict):
        raise ValueError('checkpoint must be a JSON object')
    version = document.get('schema_version')
    expected_fields = (_V1_CHECKPOINT_FIELDS if type(version) is int and version == _CHECKPOINT_SCHEMA_VERSION
                       else _V2_CHECKPOINT_FIELDS if type(version) is int and version == _INTEGRATION_CHECKPOINT_SCHEMA_VERSION
                       else _V3_CHECKPOINT_FIELDS if type(version) is int and version == _CONFIDENCE_CHECKPOINT_SCHEMA_VERSION
                       else _V4_CHECKPOINT_FIELDS if type(version) is int and version == 4
                       else None)
    if expected_fields is None or set(document) != expected_fields:
        raise ValueError('checkpoint fields do not match the frozen schema')
    return document


def _candidate_digest(candidate):
    digest = sha256(b'quality-pose-candidate-v1\0')
    for name in ('pose', 'points_object_m', 'pixels_image', 'weights'):
        try:
            array = np.asarray(getattr(candidate, name), dtype='<f8')
            digest.update(name.encode('ascii') + b'\0')
            digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
            digest.update(np.ascontiguousarray(array).tobytes(order='C'))
        except Exception:
            return None
    source = getattr(candidate, 'source', '')
    if not isinstance(source, str):
        return None
    digest.update(source.encode('utf-8') + b'\0')
    extras = getattr(candidate, 'ranker_features', None)
    if extras is not None:
        try:
            if isinstance(extras, Mapping):
                extras = {key: value for key, value in extras.items()
                          if key != 'appearance_summary_ms'}
            detached = json.dumps(extras, sort_keys=True, separators=(',', ':'),
                                  allow_nan=False, default=lambda value: np.asarray(value).tolist())
        except (TypeError, ValueError):
            detached = type(extras).__qualname__
        digest.update(detached.encode('utf-8'))
    return digest.hexdigest()


def candidate_content_ids(candidates, *, stage='candidate'):
    """Build input-order-independent IDs for valid candidate contents.

    Exact duplicates get numbered IDs; since their contents are identical,
    duplicate numbering has no bearing on non-legacy selection. ``None`` and
    malformed candidates receive stage/index IDs so each attempted slot can
    still be represented in an audit trace.
    """
    bases = []
    counts = {}
    for index, candidate in enumerate(candidates):
        base = None if candidate is None else _candidate_digest(candidate)
        if base is not None:
            counts[base] = counts.get(base, 0) + 1
        bases.append(base)
    seen = {}
    result = []
    for index, base in enumerate(bases):
        if base is None:
            result.append(f'{stage}-unavailable-{index}')
        elif counts[base] == 1:
            result.append(f'candidate-{stage}-{base[:24]}')
        else:
            duplicate_index = seen.get(base, 0)
            seen[base] = duplicate_index + 1
            result.append(f'candidate-{stage}-{base[:24]}-dup{duplicate_index}')
    return result


@dataclass(frozen=True, slots=True)
class RankCandidate:
    """One hard-gate-eligible candidate with frozen-schema feature values."""
    candidate_id: str
    original_index: int
    candidate: object
    stats: Mapping
    features: Sequence[float]
    ranker_features: Mapping | None = None

    @property
    def pose(self):
        return self.candidate.pose


@dataclass(frozen=True, slots=True)
class RankDecision:
    ordered_ids: tuple[str, ...]
    scores: Mapping[str, float]
    selected_id: str | None
    rejection_reason: str | None = None

    def __post_init__(self):
        ordered = tuple(self.ordered_ids)
        detached = {str(key): float(value) for key, value in self.scores.items()}
        if not all(math.isfinite(value) for value in detached.values()):
            raise ValueError('RankDecision scores must be finite')
        if self.selected_id is not None and self.selected_id not in ordered:
            raise ValueError('selected_id must be in ordered_ids')
        if self.rejection_reason is not None and not isinstance(self.rejection_reason, str):
            raise ValueError('rejection_reason must be text or None')
        object.__setattr__(self, 'ordered_ids', ordered)
        object.__setattr__(self, 'scores', MappingProxyType(detached))

    def as_dict(self):
        return {
            'ordered_ids': list(self.ordered_ids),
            'scores': dict(self.scores),
            'selected_id': self.selected_id,
            'rejection_reason': self.rejection_reason,
        }


def _decision_from_scored(scored, selected_id, rejection_reason=None):
    ordered = tuple(row.candidate_id for row, _ in scored)
    return RankDecision(ordered,
                        {row.candidate_id: float(score) for row, score in scored},
                        selected_id, rejection_reason)


def _rank_context(context):
    return context if isinstance(context, Mapping) else {}


def _is_distinct(first, second, context):
    from .quality_contract import distinct_pose
    class _Settings:
        recovery_rotation_degrees = float(context.get('recovery_rotation_degrees', 20.))
        recovery_translation_m = float(context.get('recovery_translation_m', .03))
    return distinct_pose(first.pose, second.pose, _Settings())


class LegacyPoseRanker:
    """Exact stable score ordering and recovery ambiguity rule from the tracker."""
    name = 'legacy'

    def rank(self, context, candidates):
        context = _rank_context(context)
        candidates = tuple(candidates)
        # Explicit original indices preserve historical stable input-order ties
        # when cached candidates are replayed in a reordered container.
        scored = sorted(((row, float(row.stats['score'])) for row in candidates),
                        key=lambda item: (-item[1], item[0].original_index))
        selected_id = scored[0][0].candidate_id if scored else None
        rejection_reason = None
        if (selected_id is not None and bool(context.get('recovery')) and
                len(scored) > 1):
            top, top_score = scored[0]
            ratio = float(context.get('recovery_ambiguity_ratio', .9))
            if any(score >= top_score * ratio and _is_distinct(top, row, context)
                   for row, score in scored[1:]):
                selected_id = None
                rejection_reason = 'ambiguous_recovery_hypotheses'
        return _decision_from_scored(scored, selected_id, rejection_reason)


def _feature_value(row, name):
    value_index = _FEATURE_INDEX[name]
    present_index = _FEATURE_INDEX[f'has_{name}']
    if len(row.features) != FEATURE_VECTOR_SHAPE[0]:
        raise ValueError('RankCandidate feature vector has the wrong frozen shape')
    present = float(row.features[present_index])
    return float(row.features[value_index]) if present >= .5 else None


class GeometricPoseRanker:
    """Fixed, non-calibrated quality ordering that honors missing indicators."""
    name = 'geometric'

    def rank(self, context, candidates):
        scored = []
        for row in candidates:
            terms = []

            def add(name, weight, transform=lambda value: value):
                value = _feature_value(row, name)
                if value is not None:
                    terms.append((weight, float(np.clip(transform(value), 0., 1.))))

            add('inlier_ratio', .24)
            add('mean_inlier_confidence', .08)
            add('spatial_support', .16, lambda value: value / .3)
            add('validation_score', .16)
            add('median_reprojection_720', .14,
                lambda value: math.exp(-max(value, 0.) / 3.))
            add('p95_reprojection_720', .10,
                lambda value: math.exp(-max(value, 0.) / 8.))
            add('rotation_from_last_valid_degrees', .06,
                lambda value: math.exp(-max(value, 0.) / 45.))
            add('translation_from_last_valid_m', .06,
                lambda value: math.exp(-max(value, 0.) / .1))
            seconds = _feature_value(row, 'seconds_since_last_valid')
            if seconds is not None:
                terms.append((.04, math.exp(-max(seconds, 0.) / 2.)))
            else:
                frames = _feature_value(row, 'frames_since_last_valid')
                if frames is not None:
                    terms.append((.04, math.exp(-max(frames, 0.) / 30.)))
            appearance = [
                _feature_value(row, key) for key in (
                    'appearance_patch_cosine_00', 'appearance_patch_cosine_01',
                    'appearance_patch_cosine_10', 'appearance_patch_cosine_11')
            ]
            available_appearance = [value for value in appearance if value is not None]
            if available_appearance:
                terms.append((.10, float(np.clip(
                    (sum(available_appearance) / len(available_appearance) + 1.) / 2.,
                    0., 1.))))
                add('appearance_valid_fraction', .04)
            total_weight = sum(weight for weight, _ in terms)
            score = (sum(weight * value for weight, value in terms) / total_weight
                     if total_weight else 0.)
            scored.append((row, score))
        scored.sort(key=lambda item: (-item[1], item[0].candidate_id))
        return _decision_from_scored(scored, scored[0][0].candidate_id if scored else None)


class LearnedPoseRanker:
    """Validated JSON dense network with calibrated probability rejection."""
    name = 'learned'

    def __init__(self, checkpoint):
        document = _snapshot_checkpoint(checkpoint)
        provenance = document.get('provenance')
        # Direct construction accepts only explicitly synthetic analytical
        # fixtures or a complete production payload. Runtime factories below
        # always require the production provenance path.
        if not (isinstance(provenance, dict) and
                provenance.get('kind') == 'synthetic_test'):
            if document['schema_version'] == _CHECKPOINT_SCHEMA_VERSION:
                _validate_calibrated_provenance(document)
            else:
                _validate_integration_provenance(document)
        if (type(document['schema_version']) is not int or
                document['schema_version'] not in (_CHECKPOINT_SCHEMA_VERSION,
                                                   _INTEGRATION_CHECKPOINT_SCHEMA_VERSION)):
            raise ValueError('unsupported ranker checkpoint schema_version')
        if (type(document['feature_schema_version']) is not int or
                document['feature_schema_version'] != FEATURE_SCHEMA_VERSION):
            raise ValueError('ranker feature schema version mismatch')
        if document['feature_names'] != list(FEATURE_NAMES):
            raise ValueError('ranker feature names do not match the frozen order')

        feature_count = FEATURE_VECTOR_SHAPE[0]
        self.mean = _immutable_array(document['mean'], (feature_count,), 'mean')
        self.scale = _immutable_array(document['scale'], (feature_count,), 'scale')
        if np.any(self.scale <= 0.):
            raise ValueError('scale values must be strictly positive')

        architecture = document['architecture']
        if architecture not in ('linear', 'mlp64_32'):
            raise ValueError('architecture must be linear or mlp64_32')
        layers_value = document['layers']
        if not isinstance(layers_value, list):
            raise ValueError('layers must be an array')
        expected_widths = ((feature_count, 1) if architecture == 'linear'
                           else (feature_count, 64, 32, 1))
        if len(layers_value) != len(expected_widths) - 1:
            raise ValueError('layer count does not match architecture')
        frozen_layers = []
        for index, item in enumerate(layers_value):
            if not isinstance(item, dict) or set(item) != {'weight', 'bias', 'activation'}:
                raise ValueError('each layer must contain weight, bias, and activation')
            out_width, in_width = expected_widths[index + 1], expected_widths[index]
            weight = _immutable_array(item['weight'], (out_width, in_width),
                                      f'layers[{index}].weight')
            bias = _immutable_array(item['bias'], (out_width,),
                                     f'layers[{index}].bias')
            activation = item['activation']
            expected_activation = ('identity' if index == len(layers_value) - 1
                                   else 'relu')
            if activation not in ('relu', 'identity') or activation != expected_activation:
                raise ValueError('hidden layers must use relu and final layer identity')
            frozen_layers.append((weight, bias, activation))
        self.layers = tuple(frozen_layers)
        self.architecture = architecture
        self.schema_version = document['schema_version']
        if self.schema_version == _CHECKPOINT_SCHEMA_VERSION:
            self.artifact_purpose = 'production_ranker_v1'
            self._selection_enabled = True
            self.calibration_unavailable_reason = None
            self.acceptance_probability = self._probability_threshold(
                document['acceptance_probability'], 'acceptance_probability')
            self.distinct_pose_margin_probability = self._probability_threshold(
                document['distinct_pose_margin_probability'],
                'distinct_pose_margin_probability')
        else:
            _validate_integration_provenance(document)
            self.artifact_purpose = document['artifact_purpose']
            self._selection_enabled = False
            self.calibration_unavailable_reason = document['calibration_unavailable_reason']
        self.provenance = _freeze_json(document['provenance'])

    @property
    def selection_enabled(self):
        """Whether this schema carries an authorized candidate-selection policy."""
        return self._selection_enabled

    @staticmethod
    def _probability_threshold(value, name):
        if type(value) not in (int, float):
            raise ValueError(f'{name} must be a finite probability')
        number = float(value)
        if not math.isfinite(number) or not 0. <= number <= 1.:
            raise ValueError(f'{name} must be between zero and one')
        return number

    @classmethod
    def from_checkpoint(cls, checkpoint):
        document = _snapshot_checkpoint(checkpoint)
        if document['schema_version'] == _CHECKPOINT_SCHEMA_VERSION:
            _validate_calibrated_provenance(document)
        else:
            _validate_integration_provenance(document)
        return cls(document)

    def _logit(self, features):
        try:
            values = np.asarray(features, dtype=np.float64)
        except Exception as exc:
            raise ValueError('features must be a finite vector') from exc
        if values.shape != FEATURE_VECTOR_SHAPE or not np.isfinite(values).all():
            raise ValueError(f'features must be finite with shape {FEATURE_VECTOR_SHAPE}')
        activation = (values - self.mean) / self.scale
        for weight, bias, kind in self.layers:
            activation = weight @ activation + bias
            if kind == 'relu':
                activation = np.maximum(activation, 0.)
        result = float(activation[0])
        if not math.isfinite(result):
            raise ValueError('ranker produced a nonfinite logit')
        return result

    def predict_logits(self, features):
        """Return one pre-sigmoid classifier logit."""
        return self._logit(features)

    @staticmethod
    def _sigmoid(logit):
        if logit >= 0.:
            return 1. / (1. + math.exp(-logit))
        exp_value = math.exp(logit)
        return exp_value / (1. + exp_value)

    def probability(self, features):
        return self._sigmoid(self._logit(features))

    def probabilities(self, candidates):
        return [self.probability(row.features) for row in candidates]

    def rank(self, context, candidates):
        candidates = tuple(candidates)
        probabilities = [self.probability(row.features) for row in candidates]
        return self.rank_with_scores(context, candidates, probabilities)

    def rank_with_scores(self, context, candidates, probabilities):
        """Run the ranker's actual decision path on an externally computed score stream.

        Export verification uses this to compare PyTorch, NumPy, and ONNX scores
        through the same disabled integration decision behavior.
        """
        context = _rank_context(context)
        candidates = tuple(candidates)
        try:
            values = np.asarray(probabilities, dtype=np.float64)
        except Exception as exc:
            raise ValueError('ranker scores must be finite probabilities') from exc
        if (values.shape != (len(candidates),) or not np.isfinite(values).all() or
                np.any((values < 0.) | (values > 1.))):
            raise ValueError('ranker scores must match candidates and be finite probabilities')
        probabilities = values.tolist()
        scored = list(zip(candidates, probabilities))
        scored.sort(key=lambda item: (-item[1], item[0].candidate_id))
        selected_id = None
        rejection_reason = None
        if not self.selection_enabled:
            rejection_reason = 'integration_only_selection_disabled'
        elif scored:
            top, top_probability = scored[0]
            if top_probability < self.acceptance_probability:
                rejection_reason = 'learned_acceptance_probability_below_threshold'
            else:
                selected_id = top.candidate_id
                if bool(context.get('recovery')):
                    distinct = [probability for row, probability in scored[1:]
                                if _is_distinct(top, row, context)]
                    strongest_distinct = max(distinct, default=None)
                    if (strongest_distinct is not None and
                            top_probability - strongest_distinct <
                            self.distinct_pose_margin_probability):
                        selected_id = None
                        rejection_reason = (
                            'learned_distinct_pose_probability_margin_below_threshold')
        return _decision_from_scored(scored, selected_id, rejection_reason)

    def to_checkpoint(self):
        common = {
            'schema_version': self.schema_version,
            'feature_schema_version': FEATURE_SCHEMA_VERSION,
            'feature_names': list(FEATURE_NAMES),
            'mean': self.mean.tolist(),
            'scale': self.scale.tolist(),
            'layers': [
                {'weight': weight.tolist(), 'bias': bias.tolist(),
                 'activation': activation}
                for weight, bias, activation in self.layers
            ],
            'architecture': self.architecture,
            'provenance': _thaw_json(self.provenance),
        }
        if self.schema_version == _CHECKPOINT_SCHEMA_VERSION:
            common['acceptance_probability'] = self.acceptance_probability
            common['distinct_pose_margin_probability'] = self.distinct_pose_margin_probability
            return common
        common.update({
            'artifact_purpose': self.artifact_purpose,
            'selection_enabled': False,
            'calibration_unavailable_reason': self.calibration_unavailable_reason,
        })
        return common


class ExperimentalOrderingRanker(LearnedPoseRanker):
    """Explicit schema-v4 development ordering, with ordinary recovery rejection.

    Available-dimension calibration is not evidence of full rotation accuracy.
    Production loading remains schema-v1 only (or nonselecting schema-v2).
    """
    name = 'ordering-research'

    def __init__(self, checkpoint):
        from .quality_ordering_research import (
            ARTIFACT_PURPOSE, POLICY, CHECKPOINT_SELECTION, canonical_hash, validate_evidence, composite_data_identity,
        )
        document = _snapshot_checkpoint(checkpoint)
        if (document['schema_version'] != 4 or document.get('artifact_purpose') != ARTIFACT_PURPOSE or
                document.get('policy') != POLICY or document.get('research_only') is not True):
            raise ValueError('ordering research requires schema4 experimental_ordering_v1')
        provenance = document.get('provenance')
        required = {'identity', 'checkpoint_sha256', 'calibration', 'calibration_split',
                    'test_read', 'completion', 'completion_sha256'}
        if not isinstance(provenance, dict) or set(provenance) != required:
            raise ValueError('ordering provenance fields differ from schema4')
        identity = provenance['identity']
        base_fields = {'manifest_sha256', 'data', 'source', 'config_sha256',
                       'feature_names', 'feature_schema_version', 'runtime'}
        extra_fields = {'purpose', 'config', 'ordering_evidence', 'ordering_evidence_sha256'}
        if not isinstance(identity, dict) or set(identity) != base_fields | extra_fields:
            raise ValueError('ordering trainer identity fields differ from schema4')
        if identity['purpose'] != ARTIFACT_PURPOSE:
            raise ValueError('ordering trainer purpose mismatch')
        from .quality_ranker_train import TrainConfig
        config = identity['config']
        try:
            validated_config = TrainConfig(**config)
            validated_config.validate()
        except (TypeError, ValueError) as exc:
            raise ValueError('ordering config malformed') from exc
        if (validated_config.architecture != document['architecture'] or
                validated_config.pairwise_weight <= 0. or
                identity['config_sha256'] != canonical_hash({'purpose': ARTIFACT_PURPOSE, 'config': config})):
            raise ValueError('ordering purpose/config identity mismatch')
        validate_evidence(identity['ordering_evidence'])
        if identity['ordering_evidence_sha256'] != canonical_hash(identity['ordering_evidence']):
            raise ValueError('ordering supervision evidence hash mismatch')
        if identity['data'] != composite_data_identity(identity['ordering_evidence']):
            raise ValueError('ordering composite data identity differs from original sources')
        if (set(identity['source']) != _TRAINER_SOURCE_FILES | {'quality_ordering_research.py'} or
                any(not _is_sha256(value) for value in identity['source'].values())):
            raise ValueError('ordering source identity mismatch')
        completion = provenance['completion']
        completion_fields = {'artifact_purpose', 'complete', 'checkpoint_selection', 'identity_sha256',
                             'selected_checkpoint_sha256', 'final_checkpoint_sha256', 'selected_model_sha256',
                             'selected_validation_objective', 'epochs_completed', 'optimizer_steps',
                             'termination_reason', 'test_read', 'dev_read'}
        if (not isinstance(completion, dict) or set(completion) != completion_fields or
                provenance['completion_sha256'] != canonical_hash(completion) or
                completion['artifact_purpose'] != ARTIFACT_PURPOSE or completion['complete'] is not True or
                completion['checkpoint_selection'] != CHECKPOINT_SELECTION or
                completion['identity_sha256'] != canonical_hash(identity) or
                completion['selected_checkpoint_sha256'] != provenance['checkpoint_sha256'] or
                not _is_sha256(completion['final_checkpoint_sha256']) or
                not _is_sha256(completion['selected_model_sha256']) or
                completion['test_read'] is not False or completion['dev_read'] is not False or
                type(completion['epochs_completed']) is not int or
                not 0 < completion['epochs_completed'] <= validated_config.epochs or
                type(completion['optimizer_steps']) is not int or completion['optimizer_steps'] <= 0 or
                completion['termination_reason'] not in ('epochs_complete', 'early_stopping') or
                (completion['termination_reason'] == 'epochs_complete' and
                 completion['epochs_completed'] != validated_config.epochs) or
                type(completion['selected_validation_objective']) not in (int, float) or
                not math.isfinite(completion['selected_validation_objective']) or
                completion['selected_validation_objective'] < 0.):
            raise ValueError('ordering completion provenance mismatch')
        # Reuse the strict calibrated provenance and dense inference checks via
        # an internal schema1 view. The public payload always remains schema4.
        dense = {key: document[key] for key in _V1_CHECKPOINT_FIELDS}
        dense['schema_version'] = 1
        production_identity = {key: identity[key] for key in base_fields}
        production_identity['source'] = {key: value for key, value in identity['source'].items()
                                         if key in _TRAINER_SOURCE_FILES}
        dense['provenance'] = {key: value for key, value in provenance.items()
                                if key not in ('completion', 'completion_sha256')}
        dense['provenance']['identity'] = production_identity
        _validate_calibrated_provenance(dense)
        super().__init__(dense)
        self.schema_version = 4
        self.artifact_purpose = ARTIFACT_PURPOSE
        self.research_only = True
        self.policy = POLICY
        self.provenance = _freeze_json(provenance)
        self._checkpoint = document

    @classmethod
    def from_checkpoint(cls, checkpoint):
        return cls(checkpoint)

    def to_checkpoint(self):
        return json.loads(json.dumps(self._checkpoint, allow_nan=False))


class ConfidenceRejectionRanker:
    """Research-only confidence veto layered after the exact legacy policy.

    The model is evaluated only for the candidate selected by the legacy
    ranker. It cannot reorder, substitute, or resurrect any candidate.
    """
    name = 'confidence-research'

    def __init__(self, checkpoint):
        document = _snapshot_checkpoint(checkpoint)
        _validate_confidence_provenance(document)
        threshold = document['acceptance_probability']
        if (type(threshold) not in (int, float) or isinstance(threshold, bool) or
                not math.isfinite(float(threshold)) or not 0. < float(threshold) < 1.):
            raise ValueError('confidence acceptance_probability must be strictly between zero and one')
        # Reuse the established dense feature and NumPy inference validator.
        dense = {
            'schema_version': 1,
            'feature_schema_version': document['feature_schema_version'],
            'feature_names': document['feature_names'],
            'mean': document['mean'], 'scale': document['scale'],
            'layers': document['layers'], 'architecture': document['architecture'],
            'acceptance_probability': float(threshold),
            'distinct_pose_margin_probability': 0.0,
            'provenance': {'kind': 'synthetic_test'},
        }
        self._model = LearnedPoseRanker(dense)
        self.acceptance_probability = float(threshold)
        self.architecture = document['architecture']
        self.schema_version = 3
        self.artifact_purpose = document['artifact_purpose']
        self.policy = document['policy']
        self.research_only = True
        self.provenance = _freeze_json(document['provenance'])
        self._checkpoint = document
        self._last_research_diagnostic = None

    @classmethod
    def from_checkpoint(cls, checkpoint):
        return cls(checkpoint)

    def probability(self, features):
        return self._model.probability(features)

    def rank_with_selected_probability(self, context, candidates, probability):
        """Apply an externally computed probability through the same veto path."""
        candidates = tuple(candidates)
        legacy = LegacyPoseRanker().rank(context, candidates)
        return self._apply_legacy_probability(legacy, probability)

    def _apply_legacy_probability(self, legacy, probability):
        if legacy.selected_id is None:
            self._last_research_diagnostic = MappingProxyType({
                'legacy_selected_id': None, 'selected_probability': None,
                'confidence_applied': False})
            return legacy
        if (type(probability) not in (int, float) or isinstance(probability, bool) or
                not math.isfinite(float(probability)) or not 0. <= float(probability) <= 1.):
            raise RuntimeError('confidence_research_error: selected probability must be finite in [0,1]')
        self._last_research_diagnostic = MappingProxyType({
            'legacy_selected_id': legacy.selected_id,
            'selected_probability': float(probability),
            'confidence_applied': True})
        if float(probability) >= self.acceptance_probability:
            return legacy
        return RankDecision(legacy.ordered_ids, legacy.scores, None,
                            'confidence_research_probability_below_threshold')

    def rank(self, context, candidates):
        candidates = tuple(candidates)
        legacy = LegacyPoseRanker().rank(context, candidates)
        if legacy.selected_id is None:
            return self._apply_legacy_probability(legacy, float('nan'))
        selected = next((row for row in candidates
                         if row.candidate_id == legacy.selected_id), None)
        if selected is None:  # defensive: the legacy decision promises membership.
            raise RuntimeError('confidence_research_error: legacy selection is absent')
        try:
            probability = self.probability(selected.features)
        except Exception as exc:
            raise RuntimeError('confidence_research_error: selected candidate scoring failed') from exc
        return self._apply_legacy_probability(legacy, probability)

    @property
    def research_diagnostic(self):
        return self._last_research_diagnostic

    def consume_research_diagnostic(self):
        """Return the latest rank-call diagnostic exactly once for its trace."""
        diagnostic = self._last_research_diagnostic
        self._last_research_diagnostic = None
        return diagnostic

    def to_checkpoint(self):
        return json.loads(json.dumps(self._checkpoint, allow_nan=False))


def make_ranker(name, checkpoint=None):
    """Construct a named ranker; learned checkpoints are compact JSON."""
    if not isinstance(name, str):
        raise ValueError('ranker name must be text')
    normalized = name.strip().lower()
    if normalized in ('legacy', 'score'):
        if checkpoint is not None:
            raise ValueError('legacy ranker does not use a checkpoint')
        return LegacyPoseRanker()
    if normalized in ('geometric', 'features', 'deterministic'):
        if checkpoint is not None:
            raise ValueError('geometric ranker does not use a checkpoint')
        return GeometricPoseRanker()
    if normalized == 'learned':
        if checkpoint is None:
            raise ValueError('learned ranker requires a checkpoint')
        document = _snapshot_checkpoint(checkpoint)
        if document.get('schema_version') == _CONFIDENCE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError('learned ranker rejects confidence research schema v3')
        if document.get('schema_version') == 4:
            raise ValueError('learned ranker rejects experimental ordering schema v4; use ordering-research explicitly')
        return LearnedPoseRanker.from_checkpoint(document)
    if normalized == 'ordering-research':
        if checkpoint is None:
            raise ValueError('ordering-research ranker requires a checkpoint')
        return ExperimentalOrderingRanker.from_checkpoint(checkpoint)
    if normalized == 'confidence-research':
        if checkpoint is None:
            raise ValueError('confidence-research ranker requires a checkpoint')
        return ConfidenceRejectionRanker.from_checkpoint(checkpoint)
    raise ValueError(f'unknown pose ranker: {name}')

