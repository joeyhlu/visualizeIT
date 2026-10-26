"""Immutable exact-production onboarding banks; preparation only, no tracking.

Only model geometry and the first ten explicitly authorized RGB-D/pose records
are opened. Scored frames and evaluator.json are never opened. Preparation may
run in <=4 fresh processes, but must not overlap any measured tracker timing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='tracking_exact_production_bank_v1'
ARRAY_NAMES=('referencePoints','referenceDescriptors','previousGray','seedPixels','seedPoints')
SOURCES=('bench/prototype_benchmark.py','bench/renderer.py','bench/ply_model.py',
         'bench/model.py','bench/obj_model.py','bench/vision.py')
MAX_NPZ_BYTES=64*1024**2
MAX_EXPANDED_BYTES=64*1024**2
MIN_FREE_BYTES=10*1024**3
DEFAULT_CACHE_BYTES=256*1024**2
ABLATION_HARNESS_SHA256='f2bb48a0ab875c84380ff7a1bbf91cd4715206c2830cd60b359413d85813f289'


class BankCacheError(ValueError):
    pass


@dataclass(frozen=True)
class BankEntry:
    arrays: tuple | None
    metadata: dict


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>16*1024**2:
        raise BankCacheError('linked/oversized JSON metadata')
    return json.loads(path.read_text())


def write_new(path,value):
    path=Path(path);raw=canonical(value)
    if path.exists():raise BankCacheError('immutable output already exists: '+str(path))
    temporary=path.with_name(path.name+'.part')
    with temporary.open('xb') as f:f.write(raw)
    temporary.rename(path)


def configure_threads():
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    from .vision import cv2
    cv2.setNumThreads(1)


def production_api():
    from .prototype_benchmark import build_bank
    from .ply_model import read_ply
    return build_bank,read_ply


def source_hashes():
    return {p:sha(ROOT/p) for p in SOURCES}


def package_versions():
    import numpy as np
    from .vision import cv2
    return dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,
                pillow=importlib.metadata.version('Pillow'))


def resource_path(dataset,spec):
    dataset=Path(dataset).resolve()
    if not isinstance(spec,dict) or set(spec)!={'path','sha256'}:
        raise BankCacheError('resource must contain exactly path/sha256')
    lexical=dataset/spec['path'];path=lexical.resolve()
    if lexical.is_symlink() or not path.is_relative_to(dataset) or not path.is_file() or sha(path)!=spec['sha256']:
        raise BankCacheError('resource path/hash mismatch: '+str(spec['path']))
    return path


def binding(dataset,object_id,*,verify_inputs=True):
    """Bind the full label-isolated inference manifest, never evaluator contents."""
    path=Path(dataset)/'inference.json';manifest=read(path)
    matches=[o for o in manifest['objects'] if o['objectId']==object_id]
    if len(matches)!=1 or len(matches[0]['onboarding'])!=10:
        raise BankCacheError('unique object and exactly ten authorized onboarding records required')
    obj=matches[0];ids=[r['frameId'] for r in obj['onboarding']]
    if len(set(ids))!=10 or ids!=sorted(ids):raise BankCacheError('onboarding order/identity differs')
    if verify_inputs:
        resource_path(dataset,obj['model'])
        for record in obj['onboarding']:
            resource_path(dataset,record['rgb']);resource_path(dataset,record['depth'])
    body=dict(inferenceManifestSHA256=sha(path),objectId=object_id,name=obj['name'],
        model=obj['model'],onboarding=obj['onboarding'],sourceHashes=source_hashes(),
        packageVersions=package_versions(),builder='bench.prototype_benchmark.build_bank unchanged',
        helperSourceSHA256=sha(__file__),
        initialization='production_mesh; ORB nfeatures=2000 per authorized view; no additional cap')
    return body,obj


def array_info(array):
    import numpy as np
    a=np.ascontiguousarray(array)
    return dict(dtype=a.dtype.str,shape=list(a.shape),bytes=a.nbytes,
        sha256=hashlib.sha256(f'{a.dtype.str}|{a.shape}|'.encode()+a.tobytes()).hexdigest())


def validate_arrays(arrays):
    import numpy as np
    if len(arrays)!=5:raise BankCacheError('expected exactly five bank arrays')
    p,d,g,x,q=arrays;n=len(p);s=len(x)
    if (p.shape!=(n,3) or p.dtype!=np.float64 or d.shape!=(n,32) or d.dtype!=np.uint8
            or g.ndim!=2 or g.dtype!=np.uint8 or min(g.shape,default=0)<1
            or x.shape!=(s,2) or x.dtype!=np.float32 or q.shape!=(s,3) or q.dtype!=np.float64
            or n>100000 or s>100000 or g.size>16*1024**2):
        raise BankCacheError('production bank array shapes/dtypes exceed the exact interface')
    if sum(a.nbytes for a in arrays)>MAX_EXPANDED_BYTES or not all(np.isfinite(a).all() for a in (p,x,q)):
        raise BankCacheError('nonfinite/oversized production bank')
    return arrays


def read_npz(path):
    """Check ZIP/NPY bounds before NumPy allocates; never allow object pickles."""
    import numpy as np
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>MAX_NPZ_BYTES:raise BankCacheError('linked/oversized bank NPZ')
    with zipfile.ZipFile(path) as z:
        infos=z.infolist()
        if len(infos)!=5 or {i.filename for i in infos}!={n+'.npy' for n in ARRAY_NAMES}:
            raise BankCacheError('unexpected/duplicate NPZ members')
        if sum(i.file_size for i in infos)>MAX_EXPANDED_BYTES:raise BankCacheError('NPZ expansion exceeds bound')
        for item in infos:
            with z.open(item) as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=65536)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f,max_header_size=65536)
                else:raise BankCacheError('unsupported NPY header')
                size=math.prod(shape)*dtype.itemsize
                if dtype.hasobject or size>MAX_EXPANDED_BYTES or size!=item.file_size-f.tell():
                    raise BankCacheError('unsafe/malformed NPY payload')
    with np.load(path,allow_pickle=False) as archive:
        arrays=tuple(archive[name].copy() for name in ARRAY_NAMES)
    validate_arrays(arrays)
    for a in arrays:a.setflags(write=False)
    return arrays


def folder(cache,object_id):
    if not isinstance(object_id,int) or isinstance(object_id,bool) or object_id<0:
        raise BankCacheError('object ID must be a nonnegative integer')
    return Path(cache)/f'object-{object_id:06d}'


def disk_guard(cache,max_cache_bytes=DEFAULT_CACHE_BYTES):
    path=Path(cache)
    while not path.exists():path=path.parent
    if shutil.disk_usage(path).free<MIN_FREE_BYTES:raise BankCacheError('10GiB disk reserve reached')
    if Path(cache).exists() and sum(p.stat().st_size for p in Path(cache).rglob('*') if p.is_file())>max_cache_bytes:
        raise BankCacheError('production bank cache byte budget reached')


def _publish(cache,object_id,bound,*,arrays=None,error=None,seconds=None,origin=None,source_npz=None):
    import numpy as np
    directory=folder(cache,object_id);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'metadata.json').exists():raise BankCacheError('immutable bank metadata already exists')
    metadata=dict(schema=SCHEMA,binding=bound,bindingSHA256=value_sha(bound),sourceBuildSeconds=seconds,
        sourceBuildTimingScope='source builder/read_ply only; excludes resource preflight, cache serialization and tracker timing',
        origin=origin or dict(kind='computed_exact_builder'),failure=error,arrays=None,artifact=None,
        helperSourceSHA256=sha(__file__),evaluatorOpened=False,scoredFramesOpened=False,
        trackingUpdates=0,threads=dict(opencv=1,blasEnvironment=1))
    if arrays is None:
        metadata.update(status='failed',anchors=None,seedViewAnchors=None,bankSHA256=None)
    else:
        validate_arrays(arrays);path=directory/'reference-bank.npz'
        if path.exists():raise BankCacheError('unfinished bank artifact needs inspection')
        temporary=path.with_name(path.name+'.part')
        if source_npz is not None:shutil.copyfile(source_npz,temporary)
        else:
            with temporary.open('xb') as f:np.savez_compressed(f,**dict(zip(ARRAY_NAMES,arrays)))
        if temporary.stat().st_size>MAX_NPZ_BYTES:raise BankCacheError('serialized bank exceeds bound')
        temporary.rename(path)
        metadata.update(status='ready' if len(arrays[0])>=12 and len(arrays[3])>=12 else 'limited',
            anchors=len(arrays[0]),seedViewAnchors=len(arrays[3]),
            bankSHA256=hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest(),
            arrays={name:array_info(a) for name,a in zip(ARRAY_NAMES,arrays)},
            artifact=dict(path='reference-bank.npz',bytes=path.stat().st_size,sha256=sha(path)))
    metadata['metadataSHA256']=value_sha(metadata)
    write_new(directory/'metadata.json',metadata)
    if arrays is not None:
        for array in arrays:array.setflags(write=False)
    return BankEntry(arrays,metadata)


def load_bank(dataset,cache,object_id):
    bound,_=binding(dataset,object_id)
    directory=folder(cache,object_id);metadata=read(directory/'metadata.json')
    unsigned={k:v for k,v in metadata.items() if k!='metadataSHA256'}
    if (metadata.get('schema')!=SCHEMA or metadata.get('metadataSHA256')!=value_sha(unsigned)
            or metadata.get('binding')!=bound or metadata.get('bindingSHA256')!=value_sha(bound)):
        raise BankCacheError('cache metadata/source/runtime/input binding differs')
    if metadata.get('status')=='failed':
        if (metadata.get('artifact') is not None or not metadata.get('failure')
                or metadata.get('arrays') is not None or metadata.get('anchors') is not None
                or metadata.get('seedViewAnchors') is not None):raise BankCacheError('invalid failed cache entry')
        return BankEntry(None,metadata)
    if metadata.get('status') not in ('ready','limited'):raise BankCacheError('unknown cache state')
    artifact=metadata['artifact']
    if artifact.get('path')!='reference-bank.npz':raise BankCacheError('cache artifact path differs')
    path=directory/artifact['path']
    if path.stat().st_size!=artifact['bytes'] or sha(path)!=artifact['sha256']:raise BankCacheError('NPZ hash/size differs')
    arrays=read_npz(path)
    if {name:array_info(a) for name,a in zip(ARRAY_NAMES,arrays)}!=metadata['arrays']:
        raise BankCacheError('bank array hashes/shapes/dtypes differ')
    legacy=hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()
    if (legacy!=metadata['bankSHA256'] or len(arrays[0])!=metadata['anchors']
            or len(arrays[3])!=metadata['seedViewAnchors']):raise BankCacheError('bank identity/count differs')
    expected='ready' if len(arrays[0])>=12 and len(arrays[3])>=12 else 'limited'
    if metadata['status']!=expected:raise BankCacheError('bank status does not match retained support')
    return BankEntry(arrays,metadata)


def adopt_ablation(dataset,cache,object_id,run):
    """Adopt only a sealed, known-source exact-production bank, never guess."""
    bound,_=binding(dataset,object_id);run=Path(run)
    seal=read(run/'inference-seal.json');freeze=read(run/'freeze.json')
    versions=bound['packageVersions'];hardware=freeze.get('hardware',{})
    versions_ok=(hardware.get('numpy')==versions['numpy'] and hardware.get('opencv')==versions['opencv']
        and hardware.get('pillow')==versions['pillow'] and str(hardware.get('python','')).split(' ')[0]==versions['python'])
    if not versions_ok:raise BankCacheError('ablation runtime binding differs')
    if (sha(run/'freeze.json')!=seal['freezeSHA256'] or freeze['sourceHashes']!=seal['sourceHashes']
            or seal['inferenceSHA256']!=bound['inferenceManifestSHA256']
            or seal['configuration'].get('initializationMode')!='production_mesh'
            or seal['sourceHashes'].get('bench/tracking_benchmark.py')!=ABLATION_HARNESS_SHA256
            or any(seal['sourceHashes'].get(p)!=h for p,h in bound['sourceHashes'].items())):
        raise BankCacheError('ablation source/input/mode/seal binding differs')
    matches=[s for s in seal['sequences'] if s['objectId']==object_id]
    if len(matches)!=1:raise BankCacheError('ablation sequence absent/duplicated')
    seq=matches[0];origin=dict(kind='adopted_sealed_ablation',inferenceSealSHA256=sha(run/'inference-seal.json'),
        sourceHarnessSHA256=ABLATION_HARNESS_SHA256,initializationMode='production_mesh')
    if 'bankArtifact' not in seq:
        if not seq.get('initializationFailure'):raise BankCacheError('ablation has neither bank nor recorded initialization failure')
        return _publish(cache,object_id,bound,error=seq['initializationFailure'],seconds=seq.get('initializationSeconds'),origin=origin)
    artifact=seq['bankArtifact'];path=resource_path(run,artifact);arrays=read_npz(path)
    if (hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()!=seq['bankSHA256']
            or len(arrays[0])!=seq['anchors'] or len(arrays[3])!=seq['seedViewAnchors']):
        raise BankCacheError('ablation bank array identity/count differs')
    return _publish(cache,object_id,bound,arrays=arrays,seconds=seq.get('initializationSeconds'),origin=origin,source_npz=path)


def build_one(dataset,cache,object_id,*,reuse_ablation=None):
    """Worker operation; use prepare_banks for resource-supervised execution."""
    configure_threads();disk_guard(cache)
    if (folder(cache,object_id)/'metadata.json').exists():return load_bank(dataset,cache,object_id).metadata
    if folder(cache,object_id).exists():raise BankCacheError('unfinished cache entry needs inspection; no automatic retry')
    if reuse_ablation is not None:return adopt_ablation(dataset,cache,object_id,reuse_ablation).metadata
    bound,obj=binding(dataset,object_id)
    directory=folder(cache,object_id);directory.mkdir(parents=True)
    builder,read_ply=production_api();start=time.perf_counter()
    try:
        scan,_,transform=read_ply(resource_path(dataset,obj['model']),obj['name'])
        arrays=builder(Path(dataset),obj['onboarding'],scan.original,transform)
        seconds=time.perf_counter()-start
        validate_arrays(arrays)
        after,_=binding(dataset,object_id)
        if after!=bound:raise BankCacheError('source/runtime/input changed during bank construction')
    except Exception as exc:
        return _publish(cache,object_id,bound,error=f'{type(exc).__name__}: {exc}',seconds=time.perf_counter()-start).metadata
    return _publish(cache,object_id,bound,arrays=arrays,seconds=seconds).metadata


def _rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    except (OSError,ValueError):pass
    return 0


def _worker_init(registry,token):
    configure_threads()
    write_new(Path(registry)/f'{os.getpid()}.json',dict(pid=os.getpid(),token=token))


def prepare_banks(dataset,cache,object_ids,*,workers=1,reuse_ablation=None,max_worker_rss=1536*1024**2,
                  max_total_rss=7*1024**3,max_seconds=3600,max_cache_bytes=DEFAULT_CACHE_BYTES):
    """Supervise preparation only. Resource failure stops all jobs; no retuning."""
    if (type(workers) is not int or not 1<=workers<=4 or not object_ids or len(set(object_ids))!=len(object_ids)
            or max_worker_rss<64*1024**2 or workers*max_worker_rss>max_total_rss
            or not 0<max_total_rss<=7*1024**3 or not 0<max_seconds<=7200):
        raise BankCacheError('invalid preparation worker/resource bounds')
    configure_threads();cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);disk_guard(cache,max_cache_bytes)
    bounds={oid:binding(dataset,oid)[0] for oid in object_ids}
    token=uuid.uuid4().hex;registry=cache/('.workers-'+token);registry.mkdir();peak=0;start=time.monotonic();reason=None
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=_worker_init,initargs=(str(registry),token))
    futures={executor.submit(build_one,str(dataset),str(cache),oid,reuse_ablation=reuse_ablation):oid for oid in object_ids}
    # CPython's ProcessPool has no public worker-handle API. Snapshot the owned
    # multiprocessing.Process handles in this pinned runtime BEFORE initialization
    # completes, so even a stalled import cannot evade RSS/time supervision.
    processes=getattr(executor,'_processes',None)
    if not isinstance(processes,dict):
        executor.shutdown(wait=False,cancel_futures=True)
        raise BankCacheError('unsupported ProcessPool implementation; no safe worker supervision')
    owned=list(processes.values())
    pending=set(futures);completed=[]
    try:
        while pending:
            done,pending=wait(pending,timeout=.25,return_when=FIRST_COMPLETED)
            for f in done:
                try:completed.append(f.result())
                except Exception as exc:
                    oid=futures[f]
                    if not (folder(cache,oid)/'metadata.json').exists():
                        completed.append(_publish(cache,oid,bounds[oid],error=f'worker_failure:{type(exc).__name__}:{exc}',
                            origin=dict(kind='preparation_worker_failure')).metadata)
            pids=[process.pid for process in owned if process.pid is not None]
            rss=[_rss(pid) for pid in pids];peak=max(peak,sum(rss))
            if any(value>max_worker_rss for value in rss):reason='per-worker RSS limit exceeded'
            if sum(rss)>max_total_rss:reason='total preparation RSS limit exceeded'
            if time.monotonic()-start>max_seconds:reason='preparation wall-time limit exceeded'
            try:disk_guard(cache,max_cache_bytes)
            except BankCacheError as exc:reason=str(exc)
            if reason:
                for process in owned:
                    if process.is_alive():process.terminate()
                for process in owned:
                    process.join(timeout=.5)
                    if process.is_alive():process.kill()
                for f in pending:f.cancel()
                break
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
    for oid in object_ids:
        if not (folder(cache,oid)/'metadata.json').exists():
            completed.append(_publish(cache,oid,bounds[oid],error=reason or 'worker exited without immutable metadata',
                origin=dict(kind='preparation_interrupted')).metadata)
    report=dict(schema='tracking_bank_preparation_supervision_v1',workers=workers,objectIds=object_ids,
        wallSeconds=time.monotonic()-start,peakSampledTotalRSSBytes=peak,reason=reason,
        maxWorkerRSSBytes=max_worker_rss,maxTotalRSSBytes=max_total_rss,maxSeconds=max_seconds,
        sampleIntervalSeconds=.25,trackingUpdates=0,threadsPerWorker=1,
        entries=[dict(objectId=oid,status=read(folder(cache,oid)/'metadata.json')['status']) for oid in object_ids])
    write_new(cache/f'preparation-{token}.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--object-ids',type=int,nargs='+',required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--reuse-ablation',type=Path)
    p.add_argument('--max-seconds',type=float,default=3600);p.add_argument('--max-worker-rss',type=int,default=1536*1024**2)
    args=p.parse_args();report=prepare_banks(args.dataset,args.cache,args.object_ids,workers=args.workers,
        reuse_ablation=str(args.reuse_ablation) if args.reuse_ablation else None,max_seconds=args.max_seconds,
        max_worker_rss=args.max_worker_rss)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
