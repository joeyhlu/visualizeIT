"""Portable v2 preparation. Labels are accessible only after numerical sealing.

No import loads Torch. Production execution uses the unchanged authenticated
ROOT18 producers in a fresh supervised interpreter, never a replacement network.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import time
import types
import uuid

GIB = 1024**3
MIB = 1024**2
MAX_JSON = 16 * MIB
FEATURE_TREE_CAP = 8 * GIB
# Deliberately conservative: uncompressed tokens/geometry/labels, receipt/index
# expansion and one atomic NPZ are covered before outcomes are observed.
PER_STATE_ENVELOPE = 16 * MIB
TEMP_ENVELOPE = 64 * MIB
ADAPTER_SHA256 = '1823fc93ef95057a3f2e4928c3d43123f130ce719683260fdce4e5f58212b388'
PILOT_SHA256 = '2fc2f3fb3b51063a39dba4f9f458e9750e2fceab33eaeab3d8172de56da4639f'
_SOURCE_MIGRATION = ContextVar('prepared_source_migration', default=None)
_ASSET_PRESENCE_SUCCESSOR = ContextVar('asset_presence_successor', default=None)
MISSING_ASSET_ADMISSION = dict(path='/kaggle/working/visualizeit-known-development-execution-admission-v1/actual-admission.json',
    bytes=5895, sha256='c8e5ebc38763e052f68aafadd913417274f7d5941efc76b4e13b96e4f4fda435')
# Sizes are recorded in the original actual-admission member of preserved
# kaggle-control-records-actual-20261018.zip (member SHA e146ff4434e03ec3fc88445a4a075885579022d4e2179343865972a06a01726c).
# Independent original producer SHA pins remain authoritative for content.
ASSET_PRESENCE_MODELS = {
    'FastSAM-x.pt': (144972346, '752cadc2828edb1cd4bc4f9eb587100631af06ea2108f4c9ed56df4755701e76'),
    'dinov2_vitl14_pretrain.pth': (1217586395, 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'),
    'gotrack_checkpoint.pt': (1608850339, 'f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086'),
    'sam2.1_hiera_base_plus.pt': (323606802, 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'),
}
ASSET_PRESENCE_SNAPSHOTS = [dict(path='/kaggle/working/visualizeit-two-view-activation-v2/branches/'+branch+'/snapshot.json',
    bytes=size, sha256=sha) for branch,size,sha in (
        ('amortized-v1',10954,'0ec3bbb086380fa74c5dd64dea2b59f85a5eca6ab8d432654cb08f116b53f033'),
        ('legacy-v1',10942,'30b5b1afc3ebc1d6798e6698af72a4a55e0eb52558aa244787bfe6e6f2a5798a'))]


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('lowercase SHA256 required')
    return value


def no_links(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute normalized path required')
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('symlink path forbidden: ' + str(path))
    return path


def loads(raw):
    def pairs(items):
        doc = {}
        for key, value in items:
            if key in doc:
                raise ValueError('duplicate JSON key: ' + key)
            doc[key] = value
        return doc
    if len(raw) > MAX_JSON:
        raise ValueError('bounded metadata size exceeded')
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('nonfinite JSON float')
        return result
    doc = json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    stack = [(doc, 0)]
    while stack:
        value, depth = stack.pop()
        if depth > 32:
            raise ValueError('JSON nesting exceeds 32')
        if isinstance(value, dict):
            stack.extend((v, depth + 1) for v in value.values())
        elif isinstance(value, list):
            stack.extend((v, depth + 1) for v in value)
    return doc


def read_json(path):
    with no_links(path).open('rb') as stream:
        return loads(stream.read(MAX_JSON + 1))


def pin(path):
    path = no_links(Path(path).absolute())
    return dict(path=path.as_posix(), bytes=path.stat().st_size, sha256=sha256(path))


def descriptor_path(value, base=None):
    if not isinstance(value, dict) or set(value) != {'path', 'bytes', 'sha256'}:
        raise ValueError('exact artifact descriptor required')
    digest(value['sha256'])
    if type(value['bytes']) is not int or value['bytes'] <= 0:
        raise ValueError('positive artifact size required')
    path = Path(value['path'])
    if base is not None:
        if path.is_absolute() or '..' in path.parts or '\\' in value['path'] or ':' in value['path']:
            raise ValueError('normalized relative payload descriptor required')
        path = Path(base) / path
    return no_links(path)


def verify_pin(value, base=None):
    path = descriptor_path(value, base)
    if not path.is_file() or path.stat().st_size != value['bytes'] or sha256(path) != value['sha256']:
        raise ValueError('artifact bytes changed: ' + str(path))
    return path


def metadata(value):
    return read_json(verify_pin(value))


def atomic_json(path, value):
    path = no_links(Path(path).absolute())
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > MAX_JSON:
        raise ValueError('metadata exceeds bound')
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    created = False
    try:
        with temporary.open('xb') as stream:
            created = True
            stream.write(raw)
        # Immutable receipts: exclusive final publication, no replace race.
        if os.name == 'nt':
            os.rename(temporary, path)  # Windows rename fails if target exists.
            created = False
        else:
            os.link(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def load_source(value, name, *, expected=None):
    """Compile authenticated bytes; untrusted .pyc is never selected."""
    if expected is not None and value['sha256'] != expected:
        raise ValueError('frozen helper source pin differs')
    path = verify_pin(value)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != value['sha256']:
        raise ValueError('source changed during read')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def load_request(path, expected):
    path = no_links(Path(path).absolute())
    if sha256(path) != digest(expected):
        raise ValueError('externally authenticated request hash differs')
    return read_json(path)


def indexed(rows, name):
    if not isinstance(rows, list) or not rows:
        raise ValueError(name + ' requires complete nonempty rows')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] in result:
            raise ValueError(name + ' duplicate/invalid state')
        result[row['id']] = row
    return result


def check_seal(doc, schema):
    if doc.get('schema') != schema or doc.get('seal_sha256') != canonical({k: v for k, v in doc.items() if k != 'seal_sha256'}):
        raise ValueError('changed ' + schema)
    return doc


def sealed(doc):
    return {**doc, 'seal_sha256': canonical(doc)}


@contextmanager
def source_migration_scope(value=None):
    """Scope an explicit, externally pinned source migration; no execution grant."""
    if value is None:
        yield
        return
    doc = metadata(value)
    if (set(doc) != {'schema', 'sources', 'contexts'}
            or doc['schema'] != 'visualizeit_prepared_source_migration_v1'
            or not isinstance(doc['sources'], dict) or not isinstance(doc['contexts'], dict)):
        raise ValueError('versioned preparation source migration required')
    token = _SOURCE_MIGRATION.set((value, doc))
    try:
        yield
    finally:
        _SOURCE_MIGRATION.reset(token)


def active_source_migration():
    value = _SOURCE_MIGRATION.get()
    return None if value is None else value[0]


def verify_prepared_source(value, root, name):
    """Verify current bytes or explicitly distinguish archived/source successors."""
    root = no_links(Path(root)); expected = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or descriptor_path(value) != expected:
        raise ValueError('preparation source path differs: ' + name)
    if pin(expected) == value:
        return verify_pin(value)
    migration = _SOURCE_MIGRATION.get()
    if migration is None:
        # Preserve the legacy verifier's exception contract for ordinary pins.
        return verify_pin(value)
    entry = None if migration is None else migration[1]['sources'].get(name)
    if entry is None:
        return verify_pin(value)
    if (not isinstance(entry, dict) or set(entry) != {'original', 'archive', 'successor'}
            or entry['original'] != value
            or any(entry['archive'].get(k) != value[k] for k in ('bytes', 'sha256'))
            or descriptor_path(entry['successor']) != expected):
        raise ValueError('archived predecessor/executing successor source migration required: ' + name)
    verify_pin(entry['archive'])
    verify_pin(entry['successor'])
    return expected


def verify_prepared_source_digest(root, name, expected):
    path = no_links(Path(root)) / name
    if sha256(path) == digest(expected):
        return path
    migration = _SOURCE_MIGRATION.get()
    entry = None if migration is None else migration[1]['sources'].get(name)
    if not isinstance(entry, dict) or entry.get('original', {}).get('sha256') != expected:
        raise ValueError('archived original preparation helper required: ' + name)
    return verify_prepared_source(entry['original'], root, name)


@contextmanager
def prepared_context_scope(context_pin):
    """Read old frozen metadata under an audited source successor, if needed."""
    from . import quality_adaptation_portable as raw
    context = raw.read_context(verify_pin(context_pin), context_pin['sha256'])
    migration = _SOURCE_MIGRATION.get()
    next_pin = None if migration is None else migration[1]['contexts'].get(context.sha256)
    if next_pin is None:
        yield context
        return
    successor = raw.read_context(verify_pin(next_pin), next_pin['sha256'])
    original, current = context.document(), successor.document()
    if {k: v for k, v in original.items() if k != 'source_files'} != {
            k: v for k, v in current.items() if k != 'source_files'}:
        raise ValueError('preparation context migration may change only source pins')
    raw.verify_sources(successor)
    for name, expected in original['source_files'].items():
        if expected == current['source_files'][name]:
            continue
        entry = migration[1]['sources'].get(name, {})
        if entry.get('original', {}).get('sha256') != expected:
            raise ValueError('preparation context lacks exact archived source: ' + name)
        verify_prepared_source(entry['original'], original['source_root'], name)
    with raw._successor_sources(context, successor, Path(original['output_root']) / 'raw'):
        yield context


def prepared_raw_state(context_pin, raw_seal_pin, *, raw_continuation=None,
                       source_migration=None, exposure_audit=None, verify_payloads=True):
    """One input-only reconstruction path for CLI, shard and label workers.

    A continuation descriptor names an already materialized view. This reader
    never resumes a capture, repairs a partial attempt, or reads a label plan.
    Legacy RAW seals retain their existing verifier and schema.
    """
    from . import quality_adaptation_portable as raw
    with source_migration_scope(source_migration):
        if raw_continuation is None:
            with prepared_context_scope(context_pin) as context:
                if exposure_audit is not None:
                    bounded_recipe(raw.frozen(context)[1], metadata(exposure_audit))
                return context, raw_metadata(context, raw_seal_pin, verify_payloads=verify_payloads)
        descriptor = metadata(raw_continuation)
        keys = {'schema', 'predecessor_context', 'successor_context', 'terminal', 'receipt',
                'adapter', 'predecessor_adapter_archive', 'view_seal'}
        if (set(descriptor) != keys or descriptor['schema'] != 'visualizeit_prepared_capture_continuation_v1'
                or descriptor['successor_context'] != context_pin or descriptor['view_seal'] != raw_seal_pin):
            raise ValueError('explicit continuation context/view descriptor differs')
        context = raw.read_context(verify_pin(context_pin), context_pin['sha256'])
        predecessor_pin = descriptor['predecessor_context']
        predecessor = raw.read_context(verify_pin(predecessor_pin), predecessor_pin['sha256'])
        if exposure_audit is not None:
            with raw._successor_sources(predecessor, context, Path(predecessor.document()['output_root']) / 'raw'):
                bounded_recipe(raw.frozen(predecessor)[1], metadata(exposure_audit))
        from . import quality_adaptation_cached_capture_v1 as cached
        state = cached.continuation_metadata(predecessor, context, descriptor['terminal'], descriptor['receipt'],
            adapter_pin=descriptor['adapter'], predecessor_adapter_archive=descriptor['predecessor_adapter_archive'])
        if (verify_pin(raw_seal_pin) != state['root'] / 'raw-seal.json'
                or metadata(raw_seal_pin) != state['seal']):
            raise ValueError('continuation preparation view pin differs from authenticated attempt')
        return context, state


def verify_prepared_neural_environment(request):
    from . import quality_adaptation_portable as raw
    with source_migration_scope(request.get('source_migration')):
        with prepared_context_scope(request['context']) as context:
            return raw.verify_neural_environment(context)


def _missing_asset_admission(value):
    if value != MISSING_ASSET_ADMISSION:
        raise ValueError('only exact missing historical asset admission may be superseded')
    path = descriptor_path(value)
    if path.exists():
        raise ValueError('asset presence successor requires historical admission to be absent')


def _asset_presence_material(partition_pin, additional_audits):
    """Authenticate historical role metadata before any current payload opens."""
    doc = metadata(partition_pin)
    if (doc.get('schema') != 'visualizeit_sharded_partition_v1'
            or doc.get('partition_sha256') != canonical({k:v for k,v in doc.items() if k!='partition_sha256'})
            or canonical(doc['full_recipe']) != doc['full_recipe_sha256']
            or canonical(doc['exposure_audit']) != doc['exposure_audit_sha256']
            or set(doc['children']) != {'smoke0','group1','group2','group3'}):
        raise ValueError('exact frozen asset presence partition required')
    pairs = [(doc['full_recipe'],doc['exposure_audit'])]
    pairs.extend((child['recipe'],child['exposure_audit']) for _,child in sorted(doc['children'].items()))
    for entry in additional_audits:
        if set(entry) != {'recipe','exposure_audit'}:
            raise ValueError('exact additional imported audit descriptors required')
        recipe,audit = metadata(entry['recipe']),metadata(entry['exposure_audit'])
        if recipe != doc['children']['smoke0']['recipe']:
            raise ValueError('additional audit may authorize only unchanged imported smoke recipe')
        pairs.append((recipe,audit))
    identities = sorted({(canonical(recipe),canonical(audit)) for recipe,audit in pairs})
    evidence = {}
    for _,audit in pairs:
        for value in audit['evidence_descriptors']:
            descriptor_path(value)
            if Path(value['path']).suffix != '.json':
                raise ValueError('historical role evidence must be metadata JSON')
            old=evidence.setdefault(value['path'],value)
            if old != value: raise ValueError('conflicting historical role evidence')
    if evidence.get(MISSING_ASSET_ADMISSION['path']) != MISSING_ASSET_ADMISSION:
        raise ValueError('exact missing admission must belong to historical evidence closure')
    _missing_asset_admission(MISSING_ASSET_ADMISSION)
    for value in evidence.values():
        if value != MISSING_ASSET_ADMISSION: metadata(value)
    for value in ASSET_PRESENCE_SNAPSHOTS: metadata(value)
    token = _ASSET_PRESENCE_SUCCESSOR.set(set(identities))
    try:
        for recipe,audit in pairs: bounded_recipe(recipe,audit)
    finally:
        _ASSET_PRESENCE_SUCCESSOR.reset(token)
    return doc, [dict(recipe_sha256=r,audit_sha256=a) for r,a in identities], sorted(evidence.values(),key=lambda v:v['path'])


def _asset_presence_file(value):
    """Hash a no-follow regular file against an independently supplied pin."""
    path = descriptor_path(value)
    before=path.stat()
    if not stat.S_ISREG(before.st_mode): raise ValueError('regular current asset required')
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    # Windows CRT fstat reports creation time differently from path stat.
    # Linux cloud admission compares every identity component.
    opened=lambda s:identity(s)[:4] if os.name=='nt' else identity(s)
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as stream:
        if opened(os.fstat(stream.fileno())) != opened(before): raise ValueError('current asset identity changed')
        actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if opened(os.fstat(stream.fileno())) != opened(before): raise ValueError('current asset changed while hashing')
    if identity(path.stat()) != identity(before) or before.st_size!=value['bytes'] or actual!=value['sha256']:
        raise ValueError('current asset bytes changed: '+str(path))


def _asset_presence_inventory(partition, source_root):
    from . import quality_adaptation_portable as raw
    recipe=partition['full_recipe']
    roles={r['physical_object_id']:r['split'] for r in raw.prospective(recipe)['global_manifest']['records']}
    meshes=[]
    for asset in recipe['assets']:
        role=roles[asset['physical_object_id']]
        if role=='test':
            if asset['object_id']!='7': raise ValueError('only sealed TEST7 metadata allowed')
            continue  # Do not even stat/hash/normalize the TEST mesh path.
        if role not in ('train','validation'): raise ValueError('current asset role must be TRAIN/VAL')
        meshes.append(dict(object_id=asset['object_id'],physical_object_id=asset['physical_object_id'],
            split=role,mesh=asset['mesh']))
    expected={'1','4','6','10','13','17','18','20','21','23','29','31','32','33'}
    if (len(meshes)!=14 or {r['object_id'] for r in meshes}!=expected
            or any(r['split']!=('validation' if r['object_id']=='4' else 'train') for r in meshes)):
        raise ValueError('exact14 TRAIN/VAL current assets and original roles required')
    models=[dict(name=name,artifact=dict(path=(Path(source_root)/'.cache/model-quality/checkpoints'/name).as_posix(),bytes=size,sha256=sha))
        for name,(size,sha) in sorted(ASSET_PRESENCE_MODELS.items())]
    return sorted(meshes,key=lambda row:int(row['object_id'])),models


def current_asset_presence_proof(partition_pin, *, additional_audits, source_root, runtime, executing_sources):
    """CPU content observation only; no model import, historical or execution claim."""
    doc,_,_=_asset_presence_material(partition_pin,additional_audits)
    meshes,models=_asset_presence_inventory(doc,source_root)
    metadata(runtime)
    for value in executing_sources.values(): verify_pin(value)
    for value in [*(row['mesh'] for row in meshes),*(row['artifact'] for row in models)]: _asset_presence_file(value)
    return dict(schema='visualizeit_current_asset_presence_proof_v1',partition=partition_pin,
        observed_at=datetime.now(timezone.utc).isoformat(),scope='current_asset_presence_only',
        historical_admission_missing=True,missing_historical_admission=MISSING_ASSET_ADMISSION,
        historical_completion_claim=False,training_executed=False,quality_claim=False,
        annotation_inputs=[],test_payload_reads=0,source_root=str(source_root),runtime=runtime,
        executing_sources=executing_sources,meshes=meshes,models=models)


def asset_presence_binding_document(partition_pin, *, additional_audits, proof, producer, review, source_migration):
    _,audits,evidence=_asset_presence_material(partition_pin,additional_audits)
    return dict(schema='visualizeit_asset_presence_successor_binding_v1',partition=partition_pin,
        additional_audits=additional_audits,audits=audits,evidence=evidence,snapshots=ASSET_PRESENCE_SNAPSHOTS,
        missing_historical_admission=MISSING_ASSET_ADMISSION,proof=proof,producer=producer,review=review,source_migration=source_migration)


@contextmanager
def asset_presence_successor_scope(value=None):
    """Explicit absent-presence-receipt successor, only inside bounded_recipe."""
    if value is None:
        yield
        return
    binding=metadata(value)
    keys={'schema','partition','additional_audits','audits','evidence','snapshots','missing_historical_admission',
        'proof','producer','review','source_migration'}
    if (set(binding)!=keys or binding['schema']!='visualizeit_asset_presence_successor_binding_v1'
            or binding['missing_historical_admission']!=MISSING_ASSET_ADMISSION
            or binding['source_migration']!=active_source_migration()):
        raise ValueError('explicit exact asset presence/source successor binding required')
    review=metadata(binding['review']); verify_pin(binding['producer'])
    if (review.get('decision')!='APPROVE' or review.get('training_approved') is not False
            or review.get('operator_sha256')!=binding['producer']['sha256']
            or review.get('source_migration')!=binding['source_migration']
            or review.get('files',{}).get('bench/quality_adaptation_portable_prepare.py')!=sha256(__file__)):
        raise ValueError('independent current asset source review required')
    doc,audits,evidence=_asset_presence_material(binding['partition'],binding['additional_audits'])
    if (binding['audits']!=audits or binding['evidence']!=evidence or binding['snapshots']!=ASSET_PRESENCE_SNAPSHOTS):
        raise ValueError('exact authorized audit/evidence/snapshot closure required')
    proof=metadata(binding['proof'])
    now=datetime.now(timezone.utc)
    observed=datetime.fromisoformat(proof['observed_at'])
    if observed.utcoffset() is None or observed.utcoffset().total_seconds()!=0 or not 0 <= (now-observed).total_seconds()<300:
        raise ValueError('fresh current asset observation required')
    sources={name:pin(Path(proof['source_root'])/name) for name in review['files']}
    meshes,models=_asset_presence_inventory(doc,proof['source_root'])
    expected=dict(schema='visualizeit_current_asset_presence_proof_v1',partition=binding['partition'],
        observed_at=proof['observed_at'],scope='current_asset_presence_only',historical_admission_missing=True,
        missing_historical_admission=MISSING_ASSET_ADMISSION,historical_completion_claim=False,
        training_executed=False,quality_claim=False,annotation_inputs=[],test_payload_reads=0,
        source_root=proof['source_root'],runtime=proof['runtime'],executing_sources=sources,meshes=meshes,models=models)
    if proof!=expected: raise ValueError('exact current TRAIN/VAL model/source presence proof required')
    metadata(proof['runtime'])
    for name,source in sources.items():
        if source['sha256']!=review['files'][name]: raise ValueError('reviewed executing source changed')
    for item in [*(row['mesh'] for row in meshes),*(row['artifact'] for row in models)]: _asset_presence_file(item)
    token=_ASSET_PRESENCE_SUCCESSOR.set({(entry['recipe_sha256'],entry['audit_sha256']) for entry in audits})
    try:
        yield
    finally:
        _ASSET_PRESENCE_SUCCESSOR.reset(token)


def bounded_recipe(recipe, exposure_audit):
    """Metadata only. The audit covers every selected declared asset/role.

    Does TRAIN contain indirect TEST information? Every known historical role,
    alias and mesh digest must be independently audited before opening assets.
    An incomplete candidate catalog is deliberately insufficient.
    """
    from . import quality_adaptation_portable as raw
    if (exposure_audit.get('schema') != 'kaggle_exposure_audit_v2'
            or exposure_audit.get('complete') is not True
            or exposure_audit.get('scope') != 'visualizeit_experimental_exposure_only'
            or exposure_audit.get('recipe_sha256') != canonical(recipe)
            or not exposure_audit.get('evidence_descriptors')):
        raise ValueError('complete externally evidenced exposure audit required')
    for value in exposure_audit['evidence_descriptors']:
        if Path(value['path']).suffix != '.json':
            raise ValueError('exposure evidence must be metadata JSON, never image/pose payload')
        successor=_ASSET_PRESENCE_SUCCESSOR.get()
        if value==MISSING_ASSET_ADMISSION and successor is not None:
            if (canonical(recipe),canonical(exposure_audit)) not in successor:
                raise ValueError('audit/recipe outside explicit asset presence successor')
            _missing_asset_admission(value)
        else:
            metadata(value)
    if recipe.get('iterations') != [0]:
        raise ValueError('first bounded v2 run requires iterations [0]')
    completion_basis = exposure_audit.get('completeness_basis')
    if completion_basis not in (None, 'selected_known_roles_over_pinned_evidence'):
        raise ValueError('unknown exposure audit completeness basis')
    if completion_basis is not None and (type(exposure_audit.get('frozen_seed')) is not int
            or exposure_audit['frozen_seed'] != recipe.get('seed')):
        raise ValueError('selected known-role audit changed its frozen split seed')
    assets = {a['physical_object_id']: a for a in recipe['assets']}
    if len(assets) != len(recipe['assets']):
        raise ValueError('selected asset physical identity/mesh declaration is duplicated')
    audit = exposure_audit.get('assets')
    if not isinstance(audit, list) or len(audit) != len(assets):
        raise ValueError('audit must cover every eligible asset exactly once')
    plan = raw.prospective(recipe)
    assignments = {r['physical_object_id']: r['split'] for r in plan['global_manifest']['records']}
    seen = set()
    role_owners = {}
    reused_validation, retained_test, prior_train = [], [], []
    for item in audit:
        obj = item.get('physical_object_id')
        if obj in seen or obj not in assets:
            raise ValueError('foreign/duplicate exposure asset')
        seen.add(obj)
        asset = assets[obj]
        if (item.get('mesh_sha256') != asset['mesh']['sha256']
                or item.get('aliases') != asset['aliases'] or item.get('eligible') is not True
                or not isinstance(item.get('historical_roles'), list)):
            raise ValueError('exposure identity/eligibility mismatch')
        roles = item['historical_roles']
        normalized = set()
        for role in roles:
            if not isinstance(role, str) or role.lower() not in ('train', 'val', 'validation', 'test'):
                raise ValueError('historically exposed asset has development/unknown role')
            normalized.add('validation' if role.lower() == 'val' else role.lower())
        if len(normalized) > 1 or (completion_basis is not None and not normalized):
            raise ValueError('mixed or missing selected known historical roles')
        if normalized:
            role = next(iter(normalized))
            if assignments[obj] != role:
                raise ValueError('historically exposed asset cannot be reassigned')
            # Identity and geometry aliases cannot declare incompatible roles.
            keys = [('mesh', asset['mesh']['sha256']), ('object', obj.lower())]
            keys += [('object', value.lower()) for value in asset['aliases']]
            for key in keys:
                if role_owners.setdefault(key, role) != role:
                    raise ValueError('alias/mesh historical-role contradiction')
            if role == 'train':
                if item.get('allow_prior_train') is not True:
                    raise ValueError('historically exposed asset requires explicit prior TRAIN reuse')
                prior_train.append(obj)
            else:
                if type(exposure_audit.get('frozen_seed')) is not int or exposure_audit['frozen_seed'] != recipe.get('seed'):
                    raise ValueError('historically exposed validation/TEST retention requires the frozen split seed')
                if role == 'validation':
                    if item.get('allow_prior_validation_reuse') is not True:
                        raise ValueError('historically exposed validation requires explicit development reuse')
                    reused_validation.append(obj)
                else:
                    if item.get('retain_prior_test_metadata_only') is not True:
                        raise ValueError('historically exposed TEST requires explicit sealed metadata-only retention')
                    retained_test.append(obj)
    active = sum(r['split'] in ('train', 'validation') for r in plan['global_manifest']['records'])
    envelope = active * PER_STATE_ENVELOPE + TEMP_ENVELOPE
    if envelope > FEATURE_TREE_CAP:
        raise ValueError('complete prospective inventory exceeds unchanged 8 GiB feature/label/temp ceiling')
    return dict(plan=plan, active_states=active, feature_tree_worst_case_bytes=envelope,
                targets=plan['target_requests'], audit_sha256=canonical(exposure_audit), acquisition_claim=False,
                exposure_disclosure=dict(prior_train_objects=sorted(prior_train),
                    reused_development_validation_objects=sorted(reused_validation),
                    retained_sealed_test_metadata_only_objects=sorted(retained_test),
                    independent_validation_claim=False, independent_test_claim=False,
                    evaluation_role='reused_development_validation_for_checkpoint_selection' if reused_validation else 'development_validation_no_quality_claim',
                    completeness_basis=completion_basis, execution_admission=False))


def raw_metadata(context, raw_seal_pin, *, verify_payloads=True):
    """Finish the complete join before hashing ANY active payload."""
    from . import quality_adaptation_portable as raw
    from .quality_adaptation_refinement_states import state_rows
    doc, spec, plan, physical = raw.frozen(context)
    root = Path(doc['output_root'])
    seal = check_seal(metadata(raw_seal_pin), 'kaggle_raw_capture_seal_v1')
    captures = read_json(root / 'raw/capture-index.json')['rows']
    receipt = read_json(root / 'raw/capture-receipt.json')
    if (seal.get('context_sha256') != context.sha256 or seal.get('prospective_plan_sha256') != sha256(root / 'scale-plan.json')
            or seal.get('physical_frames') != physical or seal.get('refinement_states') != captures
            or receipt.get('schema') != 'kaggle_raw_capture_receipt_v1'
            or receipt.get('context_sha256') != context.sha256
            or receipt.get('input_plan_sha256') != sha256(root / 'input-plan.json')
            or receipt.get('physical_frames_requested') != len(physical)
            or receipt.get('refinement_states_requested') != len(captures)
            or receipt.get('annotation_inputs') != [] or receipt.get('test_payload_reads') != 0
            or receipt.get('training_ready') is not False
            or seal.get('capture_fatal_error') != receipt.get('fatal_error')
            or seal.get('cleanup_errors') != receipt.get('cleanup_errors')
            or seal.get('training_ready') is not False or seal.get('final_execution_inventory') is not False
            or seal.get('prospective_inputs_are_recipe_identity_only') is not True
            or seal.get('annotation_inputs') != [] or seal.get('test_payload_reads') != 0):
        raise ValueError('raw seal/context/capture metadata join differs')
    expected = indexed(state_rows(dict(rows=physical)), 'prospective captures')
    actual = indexed(captures, 'actual captures')
    if set(expected) != set(actual) or len(captures) != len(plan['global_manifest']['records']):
        raise ValueError('raw capture accounting is incomplete')
    if read_json(root / 'input-plan.json') != raw.capture_plan(context, physical):
        raise ValueError('input projection changed')
    expected_artifacts = {}
    for row in physical:
        if row['split'] in ('test', 'dev'):
            continue
        for kind, keys in (('render', ('input', 'video', 'mesh')), ('bank', ('bank', 'cnos_bank')), ('mask', ('mask', 'mask_index'))):
            if row[kind]['status'] == 'available':
                for key in keys:
                    value = row[kind][key]
                    descriptor_path(value)
                    expected_artifacts[value['path']] = value
    for ident, row in actual.items():
        if any(row.get(k) != expected[ident][k] for k in ('object_id', 'split', 'sequence_id', 'capture_group', 'frame_id', 'timestamp_s', 'iteration')):
            raise ValueError('raw frozen state identity changed')
        if row['split'] in ('test', 'dev'):
            if (row.get('status') != 'unavailable' or row.get('query_rgb_sha256') is not None
                    or set(row) != set(expected[ident])):
                raise ValueError('sealed TEST/dev payload descriptors forbidden')
        if row.get('status') == 'available':
            digest(row.get('query_rgb_sha256'))
            for key in ('crops', 'capture'):
                value = row[key]
                path = descriptor_path(value, root / 'raw')
                absolute = {**value, 'path': path.as_posix()}
                expected_artifacts[absolute['path']] = absolute
        elif row.get('status') != 'unavailable' or not row.get('failure_reason'):
            raise ValueError('unavailable state must preserve reason')
    for name in ('context.json', 'recipe.json', 'scale-plan.json', 'split.json', 'freeze.json', 'physical-index.json',
                 'input-plan.json', 'raw/capture-index.json', 'raw/capture-receipt.json'):
        expected_artifacts[name] = pin(root / name)
    # The producer seals BEFORE its stage finally publishes the closing receipt.
    # That one receipt is authenticated separately, avoiding a self reference.
    for path in root.glob('stage-*.json'):
        if path.name != 'stage-seal-capture.json':
            expected_artifacts[path.name] = pin(path)
    for name in ('render', 'banks', 'masks', 'capture', 'seal-capture'):
        stage_receipt = read_json(root / ('stage-' + name + '.json'))
        if (stage_receipt.get('stage') != name or stage_receipt.get('context_sha256') != context.sha256
                or stage_receipt.get('status') != 'completed' or stage_receipt.get('error') is not None
                or stage_receipt.get('test_payload_reads') != 0 or stage_receipt.get('training_ready') is not False):
            raise ValueError('raw stage was not cleanly completed: ' + name)
    if seal.get('artifacts') != expected_artifacts or seal.get('physical_frames_requested') != len(physical) or seal.get('refinement_states_requested') != len(captures):
        raise ValueError('raw seal artifact/accounting inventory differs')
    reject_aliases([(r['split'], r.get('query_rgb_sha256'), [descriptor_path(r[k], root / 'raw') for k in ('crops', 'capture')] if r['status'] == 'available' else []) for r in captures])
    # Metadata pass has completed. The registry excludes labels and sealed rows.
    for value in expected_artifacts.values() if verify_payloads else ():
        path = descriptor_path(value)
        if not path.is_relative_to(root):
            raise ValueError('raw artifact escapes root')
        verify_pin(value)
    return dict(context=doc, recipe=spec, plan=plan, physical=physical, captures=captures, seal=seal, root=root,
                closing_stage=pin(root / 'stage-seal-capture.json'))


def reject_aliases(rows):
    owners = {}
    for split, rgb, paths in rows:
        if rgb is not None:
            key = ('rgb', rgb)
            if owners.setdefault(key, split) != split:
                raise ValueError('cross-split RGB content')
        for path in paths:
            path = no_links(path)
            keys = [('path', str(path.resolve(strict=False)))]
            # stat is metadata only; sealed payload bytes are never opened.
            if path.exists():
                info = path.stat()
                keys.append(('inode', info.st_dev, info.st_ino))
            for key in keys:
                if owners.setdefault(key, split) != split:
                    raise ValueError('cross-split payload alias')


def execution_identity(raw_state, destination):
    """Replace only inputs_sha256, retaining exact original split file bytes."""
    from . import quality_adaptation_scale as scale
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    root, plan = raw_state['root'], raw_state['plan']
    captures = indexed(raw_state['captures'], 'captures')
    physical = {p['sequence_id']: p for p in raw_state['physical']}
    records, bridge = [], []
    for state in plan['global_manifest']['records']:
        row = captures[state['id']]
        frame = physical[state['sequence_id']]
        active = state['split'] in ('train', 'validation')
        identity = dict(schema='kaggle_input_identity_v2', frozen_state={k: v for k, v in state.items() if k != 'inputs_sha256'},
            prospective_recipe_identity=state['inputs_sha256'], raw_seal_sha256=raw_state['seal']['seal_sha256'],
            crop_availability=row['status'], input_availability=frame['render']['status'] if active else 'sealed_metadata_only',
            query_rgb_sha256=row.get('query_rgb_sha256'), raw_closing_stage=raw_state['closing_stage'], annotation_inputs=[])
        if active:
            identity['source_inputs'] = {k: frame['render'].get(k) for k in ('input', 'video', 'mesh')}
            identity['capture_inputs'] = {k: row.get(k) for k in ('crops', 'capture')}
            identity['failure_reason'] = row.get('failure_reason')
        path = destination / ('identity-' + hashlib.sha256(state['id'].encode()).hexdigest() + '.json')
        atomic_json(path, identity)
        value = pin(path)
        records.append({k: v for k, v in dict(state, inputs_sha256=value['sha256']).items() if k != 'split'})
        bridge.append(dict(id=state['id'], split=state['split'], provenance=value))
    result = scale.plan_scale(dict(records=records, background_catalog=plan['global_manifest']['split_manifest']['background_catalog'],
        label_policy=plan['global_manifest']['split_manifest']['label_policy']), seed=plan['global_manifest']['seed'],
        shard_size=max((len(s['rows']) for s in plan['shards']), default=64),
        targets=[x['train_states_requested'] for x in plan['target_requests']])
    for old, new in zip(plan['global_manifest']['records'], result['global_manifest']['records']):
        if {k: v for k, v in old.items() if k != 'inputs_sha256'} != {k: v for k, v in new.items() if k != 'inputs_sha256'}:
            raise ValueError('execution provenance changed a frozen state or split')
    split_bytes = (root / 'split.json').read_bytes()
    if result['global_manifest']['split_manifest'] != plan['global_manifest']['split_manifest']:
        raise ValueError('execution split differs')
    (destination / 'split.json').write_bytes(split_bytes)
    atomic_json(destination / 'execution-plan.json', result)
    atomic_json(destination / 'identity-bridge.json', dict(schema='kaggle_identity_bridge_v2',
        raw_seal=pin(root / 'raw-seal.json'), prospective_plan=pin(root / 'scale-plan.json'), execution_plan=pin(destination / 'execution-plan.json'),
        original_split=pin(root / 'split.json'), execution_split=pin(destination / 'split.json'), rows=bridge,
        recipe_identity_is_execution_identity=False))
    return result


def verify_execution(raw_state, destination):
    from . import quality_adaptation_scale as scale
    destination = Path(destination)
    bridge = read_json(destination / 'identity-bridge.json')
    plan = scale.validate_plan(metadata(bridge['execution_plan']))
    if (bridge.get('schema') != 'kaggle_identity_bridge_v2' or bridge.get('recipe_identity_is_execution_identity') is not False
            or metadata(bridge['prospective_plan']) != raw_state['plan'] or metadata(bridge['raw_seal']) != raw_state['seal']
            or verify_pin(bridge['original_split']).read_bytes() != verify_pin(bridge['execution_split']).read_bytes()
            or metadata(bridge['execution_split']) != plan['global_manifest']['split_manifest']):
        raise ValueError('execution bridge/split changed')
    identities = indexed(bridge['rows'], 'bridge')
    old = indexed(raw_state['plan']['global_manifest']['records'], 'prospective states')
    captures = indexed(raw_state['captures'], 'captures')
    frames = {r['sequence_id']: r for r in raw_state['physical']}
    if set(identities) != set(old):
        raise ValueError('execution bridge incomplete')
    for state in plan['global_manifest']['records']:
        original = old[state['id']]
        item = identities[state['id']]
        identity = metadata(item['provenance'])
        frame, capture = frames[state['sequence_id']], captures[state['id']]
        expected = dict(schema='kaggle_input_identity_v2', frozen_state={k: v for k, v in original.items() if k != 'inputs_sha256'},
            prospective_recipe_identity=original['inputs_sha256'], raw_seal_sha256=raw_state['seal']['seal_sha256'],
            crop_availability=capture['status'], input_availability=frame['render']['status'] if state['split'] in ('train', 'validation') else 'sealed_metadata_only',
            query_rgb_sha256=capture.get('query_rgb_sha256'), raw_closing_stage=raw_state['closing_stage'], annotation_inputs=[])
        if state['split'] in ('train', 'validation'):
            expected.update(source_inputs={k: frame['render'].get(k) for k in ('input', 'video', 'mesh')},
                capture_inputs={k: capture.get(k) for k in ('crops', 'capture')}, failure_reason=capture.get('failure_reason'))
        if (identity != expected or item['split'] != state['split'] or item['provenance']['sha256'] != state['inputs_sha256']
                or {k: v for k, v in state.items() if k != 'inputs_sha256'} != {k: v for k, v in original.items() if k != 'inputs_sha256'}):
            raise ValueError('actual input provenance bridge changed')
    return plan


def crop_manifest(raw_state, destination):
    from .quality_adaptation_data import CONVENTION
    rows = []
    for row in raw_state['captures']:
        if row['split'] in ('train', 'validation') and row['status'] == 'available':
            # ROOT18 descriptor accepts only relative paths within manifest.root.
            rows.append(dict(id=row['id'], crops=row['crops']))
    if not rows:
        raise ValueError('no available input crops')
    path = raw_state['root'] / 'raw/v2-crops.json'
    atomic_json(path, dict(schema='correspondence_input_crops_v1', conventions=CONVENTION, rows=rows))
    return pin(path)


def verify_no_bytecode(root):
    root = Path(root)
    if any(any((root / directory).rglob('*.pyc')) for directory in ('bench', 'scripts')):
        raise ValueError('unverified wrapper bytecode present; require a clean source-only package')
    if any((root / '.cache/model-quality/sources').rglob('*.pyc')):
        raise ValueError('unverified upstream bytecode present; require source-only model trees')


def prepared_python(python, runtime_pin, adapter_pin, working_root):
    """Preserve the venv path; admit only its immutable runtime link proof.

    Regular paths retain the legacy no-links rule. This metadata-only check
    does not run/probe the interpreter; source_runtime still verifies the venv
    prefix, packages and neural assets before any actual producer stage.
    """
    path = Path(python).absolute()
    no_links(path.parent)
    if not path.is_symlink():
        return no_links(path)
    receipt = metadata(runtime_pin)
    recorded = receipt.get('runtime', {})
    if (receipt.get('schema') != 'kaggle_isolated_runtime_v1'
            or recorded.get('executable') != str(path)):
        raise ValueError('interpreter link must be the exact pinned runtime executable')
    adapter = load_source(adapter_pin, '_prepared_runtime_interpreter_adapter', expected=ADAPTER_SHA256)
    policy = adapter.runtime_link_policy(receipt, no_links(Path(working_root).absolute()))
    target = policy.get(path)
    if (target is None or not no_links(target).is_file()
            or recorded.get('executable_sha256') != sha256(target)):
        raise ValueError('exact authenticated runtime interpreter link and executable bytes required')
    return path


def source_runtime(request):
    k = load_source(request['adapter'], '_kaggle_v1_for_v2', expected=ADAPTER_SHA256)
    root18 = k.verify_source(no_links(Path(request['source_install'])))
    for root in (Path(request['source_root']), root18):
        verify_no_bytecode(root)
    runtime = metadata(request['runtime'])
    observed = k.verify_runtime(Path(request['python']), runtime, root18,
        (Path(request['input_root']), Path(request['working_root'])))
    k.verify_neural_assets(root18, request['neural_inputs'])
    raw_checkpoint = no_links(Path(request['source_root']) / '.cache/model-quality/checkpoints/gotrack_checkpoint.pt')
    historical_checkpoint = root18 / '.cache/model-quality/checkpoints/gotrack_checkpoint.pt'
    if not raw_checkpoint.is_file() or not os.path.samefile(raw_checkpoint, historical_checkpoint):
        raise ValueError('both fixed loaders must share the same authenticated checkpoint inode; duplicate copies forbidden')
    return k, root18, observed


def cloud_guard(request):
    if (platform.system() != 'Linux' or not os.environ.get('KAGGLE_KERNEL_RUN_TYPE')
            or os.environ.get('VISUALIZEIT_CLOUD') != '1' or not Path('/kaggle/input').is_dir()):
        raise ValueError('actual explicitly authorized Kaggle execution required')
    working = no_links(Path(request['working_root']))
    if not working.is_relative_to(Path('/kaggle/working')) or working == Path('/kaggle/working'):
        raise ValueError('dedicated Kaggle working directory required')


def quota_check(request, k, *, extra=0):
    limits = k.validate_limits(request['limits'])
    if limits['max_rss_bytes'] > 8 * GIB or limits['max_output_bytes'] > FEATURE_TREE_CAP:
        raise ValueError('v2 preparation/training exceeds 8 GiB bounds')
    working = Path(request['working_root'])
    runtime = metadata(request['runtime'])
    links = k.runtime_link_policy(runtime, working)
    roots = tuple(no_links(Path(p)) for p in request['quota_roots'])
    if working not in roots or Path(runtime['runtime_root']) not in roots:
        raise ValueError('combined quota must include working and runtime')
    required = [Path(request[key]) for key in ('source_root', 'source_install', 'features_root', 'execution_root')]
    if 'context' in request:
        raw_context = metadata(request['context'])
        required.extend(Path(raw_context[key]) for key in ('output_root', 'scratch_root', 'model_cache'))
    for path in required:
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError('combined quota omits source/model/preparation tree')
    cache_roots = (Path(request['source_root']) / '.cache/model-quality',
                   Path(request['source_install']) / k.ROOT18 / '.cache/model-quality')
    if k.unique_tree_bytes(cache_roots) > 8 * GIB:
        raise ValueError('combined inode-aware model caches exceed 8 GiB')
    resident = k.unique_tree_bytes(roots, approved_links=links)
    free = min(k.free_by_filesystem(roots).values())
    if (resident + extra + limits['reserve_bytes'] > request['storage_limit_bytes']
            or extra + limits['reserve_bytes'] > free):
        raise ValueError('observed combined quota/free reserve cannot fit complete enforced envelope')
    return dict(resident_bytes=resident, actual_free_bytes=free, reserve_bytes=limits['reserve_bytes'],
                storage_limit_bytes=request['storage_limit_bytes'], extra_envelope_bytes=extra,
                quota_scope=request['quota_roots'], inode_aware=True)


def production_run(request, module, args, *, output_root, log, cuda=False, additional_bytes=None):
    """One unchanged producer per fresh process with the authenticated bootstrap."""
    cloud_guard(request)
    k, root18, _ = source_runtime(request)
    evidence = preparation_evidence(request)
    from . import quality_adaptation_larger as larger
    argv = [request['python'], '-E', '-s', '-B', '-m', module, *map(str, args)]
    env = k.clean_environment()
    env['CUDA_VISIBLE_DEVICES'] = evidence['gpu']['uuid']
    if cuda:
        bootstrap = larger._CUDA_BOOTSTRAP
        env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', VISUALIZEIT_CUDA_BOOTSTRAP_SHA256=hashlib.sha256(bootstrap.encode()).hexdigest(),
            VISUALIZEIT_CUDA_RUNTIME_RECEIPT=str(Path(output_root) / 'device-runtime.json'))
        argv = [request['python'], '-E', '-s', '-B', '-c', bootstrap, module, *map(str, args)]
    extra = request['limits']['max_output_bytes'] if additional_bytes is None else additional_bytes
    quota_check(request, k, extra=extra)
    result = k.supervise(argv, cwd=root18, output_root=Path(output_root), working=Path(request['working_root']),
        log_path=Path(log), limits=request['limits'], env=env)
    result['observed_quota_after'] = quota_check(request, k)
    return result


def preparation_evidence(request):
    evidence = metadata(request['execution_evidence'])
    now = datetime.now(timezone.utc).timestamp()
    observed = datetime.fromisoformat(evidence['observed_at'].replace('Z', '+00:00')).timestamp()
    expires = datetime.fromisoformat(evidence['expires_at'].replace('Z', '+00:00')).timestamp()
    quota = evidence['quota']
    if (evidence.get('schema') != 'kaggle_preparation_evidence_v2' or evidence.get('provider') != 'kaggle'
            or evidence.get('raw_seal') != request['raw_seal'] or evidence.get('context') != request['context']
            or evidence.get('source_files') != request['source_files'] or not evidence.get('session_id')
            or not now - 3600 <= observed <= now < expires <= observed + 3600
            or quota.get('paid') is not False or quota.get('gpu_enabled') is not True
            or quota.get('storage_scope') != request['quota_roots'] or quota.get('storage_limit_bytes') != request['storage_limit_bytes']
            or min(quota.get('gpu_remaining_seconds', 0), quota.get('session_remaining_seconds', 0)) < request['limits']['max_seconds']):
        raise ValueError('fresh observed free Kaggle preparation quota/context/source evidence required')
    rows = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], check=True,
        capture_output=True, text=True, timeout=15).stdout.splitlines()
    if rows.count(f"{evidence['gpu']['uuid']}, {evidence['gpu']['name']}") != 1:
        raise ValueError('selected preparation GPU identity changed')
    return evidence


def run_labels(request, request_path, expected):
    """The label renderer is a fresh process under the same RSS/quota watchdog."""
    cloud_guard(request)
    from scripts.kaggle_prepared_v2 import verify_wrapper_sources
    verify_wrapper_sources(request)
    k, _, _ = source_runtime(request)
    evidence = preparation_evidence(request)
    output = Path(request['features_root'])
    quota_check(request, k, extra=TEMP_ENVELOPE)
    env = k.clean_environment()
    env.update(CUDA_VISIBLE_DEVICES=evidence['gpu']['uuid'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
        VISUALIZEIT_V2_SUPERVISED_LABEL=expected)
    result = k.supervise([request['python'], '-E', '-s', '-B', '-m', 'bench.quality_adaptation_portable_prepare',
        'label-worker', '--request', str(request_path), '--request-sha256', expected],
        cwd=Path(request['source_root']), output_root=output, working=Path(request['working_root']),
        log_path=Path(request['execution_root']) / 'label-worker.log', limits=request['limits'], env=env)
    atomic_json(Path(request['execution_root']) / 'label-stage-observation.json', result)
    return result


def extract(request, raw_state, execution):
    verify_execution(raw_state, execution)
    features = Path(request['features_root'])
    if features.exists():
        raise FileExistsError('fresh global extraction directory required')
    manifest_pin = crop_manifest(raw_state, execution)
    active_count = sum(r['split'] in ('train', 'validation') for r in raw_state['captures'])
    envelope = active_count * PER_STATE_ENVELOPE + TEMP_ENVELOPE
    if envelope > min(FEATURE_TREE_CAP, request['limits'].get('max_output_bytes', FEATURE_TREE_CAP)):
        raise ValueError('global feature/label/temp envelope exceeds unchanged extractor ceiling')
    result = None
    try:
        with preparation_lock(raw_state):
            result = production_run(request, 'bench.quality_adaptation_prepare', ['--manifest', manifest_pin['path'], '--output', features,
                '--device', 'cuda', '--max-seconds', int(request['limits']['max_seconds']), '--max-output-bytes', FEATURE_TREE_CAP,
                '--max-rows', active_count, '--cloud-execution'], output_root=features, log=Path(execution) / 'extract.log', cuda=True,
                additional_bytes=envelope)
    finally:
        if features.is_dir() and (features / 'index.json').is_file():
            # Exact original receipts survive. Only the reviewed final index adds
            # captured-unavailable and sealed TEST accounting; dev stays separate.
            for name in ('recipe.json', 'index.json'):
                shutil.copyfile(features / name, features / ('original-' + name))
            original = read_json(features / 'index.json')
            actual = indexed(original['rows'], 'original extractor')
            required = {r['id'] for r in raw_state['captures'] if r['status'] == 'available' and r['split'] in ('train', 'validation')}
            if set(actual) != required:
                raise ValueError('unchanged extractor omitted requested accounting')
            final_rows = []
            for row in raw_state['captures']:
                if row['split'] == 'dev':
                    continue
                final_rows.append(actual.get(row['id'], dict(id=row['id'], status='unavailable', failure_reason=row.get('failure_reason') or 'capture_unavailable')))
            final = {**original, 'rows': final_rows}
            # The ONLY permitted replacement is the final explicitly audited index.
            temporary = features / '.final-index.json'
            atomic_json(temporary, final)
            os.replace(temporary, features / 'index.json')
            atomic_json(features / 'extraction-binding.json', dict(schema='kaggle_extraction_binding_v2',
                original_recipe=pin(features / 'original-recipe.json'), original_index=pin(features / 'original-index.json'),
                final_recipe=pin(features / 'recipe.json'), final_index=pin(features / 'index.json'), crop_manifest=manifest_pin,
                execution_plan=pin(Path(execution) / 'execution-plan.json'), raw_seal=pin(raw_state['root'] / 'raw-seal.json'),
                unavailable_added=[r['id'] for r in final_rows if r['id'] not in actual], dev_rows=[r for r in raw_state['captures'] if r['split'] == 'dev'],
                execution_result=result, successful_original_rows_unchanged=True))
    return result


@contextmanager
def preparation_lock(raw_state):
    from .quality_adaptation_larger import stage_lock
    with stage_lock():
        yield


def prelabel_material(raw_state, execution, features):
    """Numerically reconstruct every good crop before FIRST generation open."""
    import numpy as np
    from .quality_adaptation_data import CROP_KEYS, FEATURE_KEYS, HOOKS, CONVENTION, crop_rgb_hash, load_npz, validate_crops, validate_features
    from .quality_adaptation_refinement_states import GEOMETRY_KEYS
    from .quality_adaptation_prepare import crops_from_retained_capture
    from .quality_adaptation_expanded import _check_parity, validate_retained_geometry
    from .quality_adaptation_portable import SOURCE_LOCK_SHA256
    features = Path(features)
    plan = verify_execution(raw_state, execution)
    binding = read_json(features / 'extraction-binding.json')
    recipe = metadata(binding['original_recipe'])
    original = metadata(binding['original_index'])
    final = metadata(binding['final_index'])
    crop_doc = metadata(binding['crop_manifest'])
    lock = read_json(Path(raw_state['context']['source_root']) / 'bench/runtime/adaptation-sources.json')
    if (binding.get('schema') != 'kaggle_extraction_binding_v2' or metadata(binding['final_recipe']) != recipe
            or sha256(Path(raw_state['context']['source_root']) / 'bench/runtime/adaptation-sources.json') != SOURCE_LOCK_SHA256
            or recipe.get('producer_sha256') != lock['gotrack']['files']['bench/quality_adaptation_prepare.py']
            or recipe.get('source_lock_sha256') != SOURCE_LOCK_SHA256
            or recipe.get('source_files_sha256') != canonical(lock['gotrack']['files'])
            or recipe.get('upstream_revision') != lock['gotrack']['revision'] or recipe.get('hooks') != list(HOOKS)
            or recipe.get('annotation_inputs') != [] or recipe.get('device') != raw_state['context']['device']
            or recipe.get('crop_manifest_sha256') != canonical(crop_doc)
            or recipe.get('rgb_encoding') != 'uint8_div255_or_exact_float32_0_1_v1'
            or original.get('recipe_sha256') != canonical(recipe) or final.get('recipe_sha256') != canonical(recipe)):
        raise ValueError('original extractor receipt/producer/recipe binding changed')
    captures = indexed(raw_state['captures'], 'capture')
    extracted = indexed(final['rows'], 'final extraction')
    originals = indexed(original['rows'], 'original extraction')
    nondev = {k for k, r in captures.items() if r['split'] != 'dev'}
    required = {k for k, r in captures.items() if r['split'] in ('train', 'validation') and r['status'] == 'available'}
    if (set(extracted) != nondev or set(originals) != required
            or any(extracted[k] != v for k, v in originals.items())
            or crop_doc != dict(schema='correspondence_input_crops_v1', conventions=CONVENTION,
                rows=[dict(id=r['id'], crops=r['crops']) for r in raw_state['captures'] if r['id'] in required])
            or binding.get('unavailable_added') != [r['id'] for r in final['rows'] if r['id'] not in originals]
            or binding.get('dev_rows') != [r for r in raw_state['captures'] if r['split'] == 'dev']
            or metadata(binding['execution_plan']) != plan or metadata(binding['raw_seal']) != raw_state['seal']):
        raise ValueError('global capture/extractor/full-accounting join changed')
    if recipe['device'] == 'cuda':
        from .quality_adaptation_larger import _CUDA_BOOTSTRAP
        actual = read_json(features / 'device-runtime.json')
        expected = dict(schema='gotrack_larger_cuda_runtime_v1', device='cuda', bootstrap_sha256=hashlib.sha256(_CUDA_BOOTSTRAP.encode()).hexdigest(),
            torch_version='2.5.1+cu124', cuda_version='12.4', deterministic_algorithms=True, matmul_allow_tf32=False,
            cudnn_allow_tf32=False, cudnn_benchmark=False, cudnn_deterministic=True, cublas_workspace_config=':4096:8', threads=1)
        if actual != expected:
            raise ValueError('actual CUDA extraction policy differs')
    material = dict(schema='kaggle_prelabel_seal_v2', raw_seal=pin(raw_state['root'] / 'raw-seal.json'),
        execution_plan=pin(Path(execution) / 'execution-plan.json'), identity_bridge=pin(Path(execution) / 'identity-bridge.json'),
        binding=pin(features / 'extraction-binding.json'), recipe=pin(features / 'recipe.json'), index=pin(features / 'index.json'),
        crop_manifest=binding['crop_manifest'], accounting=raw_state['captures'], parity=[], crop_artifacts=[], feature_artifacts=[],
        annotation_inputs=[], test_payload_reads=0)
    for row in raw_state['captures']:
        if row['split'] not in ('train', 'validation') or row['status'] != 'available':
            continue
        crop_path = verify_pin(row['crops'], raw_state['root'] / 'raw')
        captured_path = verify_pin(row['capture'], raw_state['root'] / 'raw')
        crops = load_npz(crop_path, CROP_KEYS)
        actual = load_npz(captured_path, GEOMETRY_KEYS)
        validate_crops(crops)
        validate_retained_geometry(actual['crop_from_orig'])
        reconstructed = crops_from_retained_capture(dict(query_rgb_crop=crops['query_rgb'].transpose(2, 0, 1)[None],
            template_rgb=crops['template_rgb'].transpose(2, 0, 1)[None], rendered_depth_mm=actual['rendered_depth_mm'],
            rendered_mask=actual['rendered_mask'], crop_k=actual['crop_k'],
            current_crop_camera_from_object_mm=actual['current_crop_camera_from_object_mm']), preserve_float_rgb=True)
        if (any(not np.array_equal(crops[k], reconstructed[k]) for k in CROP_KEYS)
                or not np.array_equal(crops['query_K'], actual['crop_k'])
                or not np.array_equal(actual['native_k'], row['native_calibration'])
                or crop_rgb_hash(crops['query_rgb']) != row['query_rgb_sha256']
                or row.get('sampling_map', {}).get('integer_centers') != 280**2
                or not 0 <= row.get('sampling_map', {}).get('max_native_pixel_difference', -1) < .001):
            raise ValueError('retained RGB/geometry/native/sampling parity changed')
        material['crop_artifacts'].extend([pin(crop_path), pin(captured_path)])
        feature = extracted[row['id']]
        if feature['status'] == 'available':
            if feature.get('crops') != row['crops'] or feature.get('query_rgb_sha256') != row['query_rgb_sha256']:
                raise ValueError('extracted input crop identity changed')
            feature_path = verify_pin(feature['features'], features)
            values = load_npz(feature_path, FEATURE_KEYS)
            validate_features(values)
            if any(not np.array_equal(values[k], crops[k]) for k in ('template_mask', 'template_xyz_m', 'query_K')):
                raise ValueError('feature geometry differs from retained crop')
            material['parity'].append(dict(id=row['id'], **_check_parity(actual, values, crops['template_mask'])))
            material['feature_artifacts'].append(pin(feature_path))
        elif feature['status'] != 'unavailable' or not feature.get('failure_reason'):
            raise ValueError('feature failure accounting missing')
    if not material['parity']:
        raise ValueError('no successful numerical capture/extraction parity; cannot seal')
    return sealed(material)


def seal_features(raw_state, execution, features):
    material = prelabel_material(raw_state, execution, features)
    atomic_json(Path(execution) / 'prelabel-seal.json', material)
    return material


def verify_prelabel(raw_state, execution, features):
    seal = read_json(Path(execution) / 'prelabel-seal.json')
    if seal != prelabel_material(raw_state, execution, features):
        raise ValueError('prelabel seal/crop/features/parity changed')
    return seal


def labels(request, raw_state, execution):
    """Separate renderer supervision stage; no GT data enters input identities."""
    cloud_guard(request)
    features = Path(request['features_root'])
    # This statement MUST precede even reading the label plan descriptor bytes.
    prelabel = verify_prelabel(raw_state, execution, features)
    label_plan = metadata(request['label_plan'])
    if set(label_plan) != {'schema', 'split_manifest', 'generations'} or label_plan['schema'] != 'kaggle_label_plan_v2':
        raise ValueError('invalid separate label plan')
    if verify_pin(label_plan['split_manifest']).read_bytes() != (raw_state['root'] / 'split.json').read_bytes():
        raise ValueError('label split bytes differ')
    generations = {r['sequence_id']: r['generation'] for r in label_plan['generations']}
    active = {r['sequence_id'] for r in raw_state['physical'] if r['split'] in ('train', 'validation') and r['render']['status'] == 'available'}
    if len(generations) != len(label_plan['generations']) or set(generations) != active:
        raise ValueError('generation metadata inventory incomplete/foreign/sealed')
    for value in generations.values():
        descriptor_path(value)  # complete metadata join before first generation hash
    from .quality_adaptation_data import CROP_KEYS, load_npz
    from .quality_adaptation_refinement_states import GEOMETRY_KEYS
    from .quality_adaptation_prepare import project_supervision
    from .quality_adaptation_smoke import render_query_labels
    from .quality_adaptation_larger import _validate_generation
    from .quality_adaptation_runtime import atomic_stream
    from .quality_gotrack import TexturedRenderer
    from .quality_runner import read_input
    from .quality_ranker_data import read_split_manifest
    import numpy as np
    split_path = verify_pin(label_plan['split_manifest'])
    split = read_split_manifest(split_path)
    k, _, _ = source_runtime(request)
    extracted = indexed(read_json(features / 'index.json')['rows'], 'features')
    physical = {r['sequence_id']: r for r in raw_state['physical']}
    inventory, outcomes = [], []
    started = time.monotonic()
    with preparation_lock(raw_state):
        from .quality_adaptation_larger import configure_torch, cleanup_device
        torch = configure_torch('cuda', deterministic=True)
        try:
            for ordinal, source in enumerate(raw_state['captures']):
                if source['split'] == 'dev':
                    continue
                row = dict(id=source['id'], object_id=source['object_id'], sequence_id=source['sequence_id'], capture_group=source['capture_group'],
                    split='val' if source['split'] == 'validation' else source['split'], query_rgb_sha256=source.get('query_rgb_sha256'), status='unavailable',
                    failure_reason=source.get('failure_reason') or extracted[source['id']].get('failure_reason') or 'label_not_processed')
                if row['query_rgb_sha256'] is None:
                    row['unavailable_provenance'] = source.get('unavailable_provenance', 'frozen state; no retained crop')
                inventory.append(row)
                renderer = None
                if source['split'] not in ('train', 'validation') or source['status'] != 'available' or extracted[source['id']]['status'] != 'available':
                    continue
                try:
                    if time.monotonic() - started >= request['limits']['max_seconds']:
                        raise TimeoutError('label deadline exhausted')
                    quota_check(request, k, extra=TEMP_ENVELOPE)
                    frame = physical[source['sequence_id']]
                    render = frame['render']
                    bundle = verify_pin(render['input']).parent
                    manifest = read_input(bundle)
                    # Original validator expects legacy val spelling only.
                    physical_projection = dict(frame, split=row['split'], **{key: render[key] for key in ('input', 'video', 'mesh')})
                    renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'], unlit=True, disable_multisampling=True, coordinate_mode='integer_centers_v1')
                    pose = _validate_generation(metadata(generations[source['sequence_id']]), physical_projection, manifest, split, sha256(split_path), renderer.vertices_m)
                    crops = load_npz(verify_pin(source['crops'], raw_state['root'] / 'raw'), CROP_KEYS)
                    geometry = load_npz(verify_pin(source['capture'], raw_state['root'] / 'raw'), GEOMETRY_KEYS)
                    query_pose, depth, support = render_query_labels(renderer, object_id=manifest['object_id'], crop_k=crops['query_K'], crop_from_orig=geometry['crop_from_orig'], native_pose=pose)
                    target = project_supervision(template_xyz_m=crops['template_xyz_m'], template_mask=crops['template_mask'], query_K=crops['query_K'],
                        query_T_cam_from_object=query_pose, observed_depth_m=depth, observed_object_mask=support, depth_tolerance_m=.002)
                    name = f'labels-{ordinal:06d}.npz'
                    with atomic_stream(features, features / name, max_file_bytes=16 * MIB, max_output_bytes=FEATURE_TREE_CAP) as stream:
                        np.savez(stream, **target)
                    row.pop('failure_reason', None)
                    row.update(status='available', labels={**pin(features / name), 'path': name})
                except Exception as exc:
                    row['failure_reason'] = f'label_failure: {type(exc).__name__}: {exc}'[:1024]
                finally:
                    if renderer is not None:
                        renderer.close()
                    outcomes.append(dict(id=source['id'], label_status=row['status'], failure_reason=row.get('failure_reason')))
        finally:
            cleanup_device('cuda', torch)
            # All requested IDs survive even an interrupted label stage.
            seen = {r['id'] for r in inventory}
            for source in raw_state['captures']:
                if source['split'] != 'dev' and source['id'] not in seen:
                    row = dict(id=source['id'], object_id=source['object_id'], sequence_id=source['sequence_id'], capture_group=source['capture_group'],
                        split='val' if source['split'] == 'validation' else source['split'], query_rgb_sha256=source.get('query_rgb_sha256'),
                        status='unavailable', failure_reason='label_stage_interrupted')
                    if row['query_rgb_sha256'] is None:
                        row['unavailable_provenance'] = 'frozen state; no crop materialized'
                    inventory.append(row)
            atomic_json(features / 'labels.json', dict(schema='correspondence_label_inventory_v1', label_origin='synthetic_renderer_training_only', rows=inventory))
    result = production_run(request, 'bench.quality_adaptation_manifest', ['--extraction', features, '--labels', features / 'labels.json', '--output', features / 'manifest.json'],
        output_root=features, log=Path(execution) / 'assemble.log', additional_bytes=TEMP_ENVELOPE)
    if result['returncode'] != 0:
        raise ValueError('unchanged global assembler failed; all labels/failures retained')
    receipt = dict(schema='kaggle_label_receipt_v2', prelabel_seal=pin(Path(execution) / 'prelabel-seal.json'), label_plan=request['label_plan'],
        inventory=pin(features / 'labels.json'), dataset=pin(features / 'manifest.json'), outcomes=outcomes,
        label_artifacts=[dict(id=r['id'], **r['labels']) for r in inventory if r['status'] == 'available'],
        labels_entered_inference=False, test_payload_reads=0, independent_quality_claim=False, elapsed_seconds=time.monotonic() - started,
        parity_rows=len(prelabel['parity']), assemble_result=result)
    atomic_json(features / 'label-receipt.json', receipt)
    return receipt


def prepare_stage(stage, request):
    from scripts.kaggle_prepared_v2 import verify_wrapper_sources
    verify_wrapper_sources(request)
    context, state = prepared_raw_state(request['context'], request['raw_seal'],
        raw_continuation=request.get('raw_continuation'), source_migration=request.get('source_migration'),
        exposure_audit=request['exposure_audit'])
    execution = Path(request['execution_root'])
    if stage == 'identity':
        # Exposure evidence is metadata; verify it before selecting/opening any
        # source asset. Raw stages independently retain their frozen inventory.
        bounded_recipe(state['recipe'], metadata(request['exposure_audit']))
        return execution_identity(state, execution)
    if stage == 'extract':
        return extract(request, state, execution)
    if stage == 'seal-features':
        return seal_features(state, execution, Path(request['features_root']))
    if stage == 'label-worker':
        verify_prepared_neural_environment(request)
        return labels(request, state, execution)
    raise ValueError('unknown preparation stage')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('recipe-preflight', 'identity', 'extract', 'seal-features', 'labels', 'label-worker'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    request = load_request(args.request, args.request_sha256)
    if args.stage == 'recipe-preflight':
        result = bounded_recipe(metadata(request['recipe']), metadata(request['exposure_audit']))
    elif args.stage == 'labels':
        result = run_labels(request, args.request, args.request_sha256)
    else:
        if args.stage == 'label-worker' and os.environ.get('VISUALIZEIT_V2_SUPERVISED_LABEL') != args.request_sha256:
            raise ValueError('label worker requires the authenticated supervised invocation')
        result = prepare_stage(args.stage, request)
    print(json.dumps(result, allow_nan=False))
    return 3 if result.get('status') == 'incomplete' else 2 if result.get('status') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
