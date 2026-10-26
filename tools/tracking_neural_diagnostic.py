"""Bounded offline GoTrack model/refinement diagnostic, NOT serving admission.

Uses two explicitly authorized onboarding frames, previous-frame annotated seeds
and onboarding masks. This is oracle-initialized same-input model research,
not an automatic detector, tracking benchmark, held-out accuracy or phone test.
No evaluator manifest is opened, no optimization/training step is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = ('reference_best', 'adapted_export', 'frozen_baseline')
MAX_OUTPUT = 512 * 1024**2
DISK_RESERVE = 10 * 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 4 * 1024**2:
        raise ValueError('diagnostic JSON exceeds 4MiB')
    path.write_bytes(raw)


def check_disk(output):
    if shutil.disk_usage(ROOT).free < DISK_RESERVE:
        raise ValueError('less than 10GiB free disk reserve')
    if sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file()) > MAX_OUTPUT:
        raise ValueError('diagnostic output exceeds 512MiB')


def resource(dataset, ref):
    root = Path(dataset).resolve()
    p = (root / ref['path']).resolve()
    if not p.is_relative_to(root) or p.is_symlink() or sha(p) != ref['sha256']:
        raise ValueError('input hash/path mismatch')
    return p


def runtime():
    names = ('torch','torchvision','numpy','Pillow','opencv-python-headless','pyrender','trimesh','PyOpenGL')
    return dict(python=sys.version, platform=platform.platform(),
                packages={n: importlib.metadata.version(n) for n in names})


def authenticate():
    from bench import workbench_neural as wn
    closure = wn.source_closure()
    if closure['historical_missing'] or closure['unreviewed_replacements'] or not closure['review_evidence_present']:
        raise ValueError('exact source closure/review evidence unavailable')
    model_dir = ROOT / wn.DEFAULT_MODEL_DIR
    artifacts = {
        'base': (ROOT / wn.BASE_CHECKPOINT_RELATIVE, wn.PRIMARY['base_checkpoint_bytes'], wn.PRIMARY['base_checkpoint_sha256']),
        'head': (model_dir / 'head.pt', wn.PRIMARY['export_bytes'], wn.PRIMARY['export_sha256']),
        'best': (model_dir / 'best.pt', wn.PRIMARY['reference_checkpoint_bytes'], wn.PRIMARY['reference_checkpoint_sha256']),
    }
    for key, (path, size, digest) in artifacts.items():
        if path.is_symlink() or path.stat().st_size != size or sha(path) != digest:
            raise ValueError(f'{key} artifact differs from authenticated bytes')
    return artifacts, closure


def prepare(dataset, output):
    import numpy as np
    import trimesh
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(dataset) / 'inference.json'
    manifest = read(manifest_path)
    obj = manifest['objects'][0]
    if len(obj['onboarding']) != 10:
        raise ValueError('exact ten-frame authorized onboarding required')
    model_path = resource(dataset, obj['model'])
    mesh = trimesh.load(model_path, process=False, force='mesh')
    mesh.vertices = np.asarray(mesh.vertices, dtype=float) * .001
    glb = output / 'metric-model.glb'
    glb.write_bytes(mesh.export(file_type='glb'))
    if glb.stat().st_size > 64 * 1024**2:
        raise ValueError('model GLB exceeds 64MiB')
    cases = []
    for index in (8, 9):
        record, seed = obj['onboarding'][index], obj['onboarding'][index-1]
        pose = np.eye(4)
        pose[:3,:3] = np.asarray(seed['evaluationPose']['cam_R_m2c']).reshape(3,3)
        pose[:3,3] = np.asarray(seed['evaluationPose']['cam_t_m2c']) * .001
        rgb = resource(dataset, record['rgb'])
        mask = resource(dataset, record['onboardingMask'])
        cases.append(dict(caseId=f'authorized-onboarding-{record["frameId"]}', frameId=record['frameId'],
            rgbPath=str(rgb), rgbSHA256=sha(rgb), maskPath=str(mask), maskSHA256=sha(mask),
            K=record['calibration']['cam_K'], seedPoseMetres=pose.tolist(), seedFrameId=seed['frameId'],
            seedSource='authorized previous onboarding pose; identical across variants',
            maskSource='authorized BOP onboarding visible-mask annotation; NOT automatic segmentation'))
    artifacts, closure = authenticate()
    plan = dict(schema='gotrack_offline_model_diagnostic_v1', dataset=manifest['dataset'],
        inferenceManifestSHA256=sha(manifest_path), objectId=obj['objectId'], name=obj['name'],
        originalModelSHA256=obj['model']['sha256'], glbPath=str(glb.resolve()), glbSHA256=sha(glb),
        modelUnits='metres; exact source-object axes preserved; original vertex colors',
        cases=cases, variants=list(VARIANTS), iterationsPerFrame=5,
        artifactHashes={k: d for k,(_,_,d) in artifacts.items()},
        sourceClosureSHA256=closure['closure_sha256'], adapterSHA256=sha(__file__),
        settings=dict(device='cpu', threads=1, seed=17, geometry='production GoTrackRefiner legacy default',
            renderer='production TexturedRenderer legacy coordinates, unlit=True, disable_multisampling=True',
            referenceExportTolerance=dict(atol=1e-6,rtol=0.)),
        evidenceScope='Full RGB->DINOv2->decoder->DPT->PnP offline model/refinement diagnostic only',
        evaluatorInputsRead=False, trainingSteps=0, servingAdmissionGranted=False,
        caveats=['Two already-authorized onboarding images, not held-out evaluation',
                 'Known prior-frame poses and annotation masks; no automatic discovery/segmentation',
                 'Same-input independent refinement, not causal video tracking or reacquisition',
                 'Legacy geometry is not workbench_integer_metric_v1 serving admission',
                 'Instrumentation and software rendering timings are not live FPS'])
    write(output / 'plan.json', plan)
    check_disk(output)
    print(json.dumps(dict(plan=str(output/'plan.json'),cases=len(cases),sourceClosure=closure['closure_sha256'])))


@contextmanager
def safe_base_loader(torch, base):
    """Security-only scope: unchanged production loader cannot request pickle.

    Fail closed; there is no weights_only=False retry or additional safe-global
    allowlist. The official base bytes and only permitted input path are pinned.
    """
    original = torch.load
    def safe(file, *args, **kwargs):
        if not isinstance(file, (str, Path)) or Path(file).resolve() != Path(base).resolve():
            raise ValueError('unexpected production-loader checkpoint path')
        kwargs['weights_only'] = True
        kwargs['map_location'] = 'cpu'
        return original(file, *args, **kwargs)
    torch.load = safe
    try:
        yield
    finally:
        torch.load = original


def array_info(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes()).hexdigest()
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=h)


def worker(plan_path, output):
    import numpy as np
    import torch
    from PIL import Image
    from bench import quality_gotrack as qg, quality_gotrack_inference as installer
    from bench import quality_adaptation_export_parity as parity, workbench_neural as wn
    from bench.quality_contract import Frame
    from bench.vision import cv2
    output = Path(output)
    plan = read(plan_path)
    if sha(__file__) != plan['adapterSHA256']:
        raise ValueError('diagnostic adapter changed since preparation')
    artifacts, closure = authenticate()
    if closure['closure_sha256'] != plan['sourceClosureSHA256'] or runtime()['packages']['torch'] != '2.5.1+cpu':
        raise ValueError('prepared source/runtime differs')
    for case in plan['cases']:
        if sha(case['rgbPath']) != case['rgbSHA256'] or sha(case['maskPath']) != case['maskSHA256']:
            raise ValueError('prepared RGB/mask differs')
    if sha(plan['glbPath']) != plan['glbSHA256']:
        raise ValueError('prepared metric model differs')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    record, record_sha = parity.read_manifest(ROOT / wn.RECORD_RELATIVE, wn.RECORD_SHA256)
    context = parity.context_for_export(record, record_sha, artifacts['head'][0])
    all_variants = {}
    write(output / 'freeze.json',dict(planSHA256=sha(plan_path),runtime=runtime(),sourceClosure=closure,
        actualTorchThreads=torch.get_num_threads(),gpuUsed=False,safeWeightsOnly=True,
        operatorNote='No production core edits, historical receipts or serving admission fabricated'))
    for variant in VARIANTS:
        check_disk(output)
        torch.manual_seed(17);np.random.seed(17);cv2.setRNGSeed(17)
        start = time.monotonic()
        with safe_base_loader(torch, artifacts['base'][0]):
            network = qg.load_network('cpu')
        network.eval().requires_grad_(False)
        provenance = dict(role=variant,baseSHA256=artifacts['base'][2])
        if variant == 'reference_best':
            raw = artifacts['best'][0].read_bytes()
            if hashlib.sha256(raw).hexdigest()!=artifacts['best'][2]:
                raise ValueError('reference checkpoint changed before safe deserialization')
            reference = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            installer._validate_state_dict(reference['head'], network.pose_head.state_dict(), torch)
            network.pose_head.load_state_dict(reference['head'],strict=True)
            provenance['referenceSHA256']=artifacts['best'][2]
            del reference,raw
        elif variant == 'adapted_export':
            network, installed = installer.install_adapted_head(network,context,device='cpu',torch_module=torch)
            provenance.update(installed)
        provenance['installedHeadStateSHA256']=wn.module_state_sha256(network.pose_head)
        rows=[]; counts=dict(backbone=0,decoder=0,pose_head=0);handles=[]
        for name in counts:
            def count(_m,_i,_o,n=name):counts[n]+=1
            handles.append(getattr(network,name).register_forward_hook(count))
        renderer=qg.TexturedRenderer(Path(plan['glbPath']),plan['objectId'],unlit=True,disable_multisampling=True)
        refiner=qg.GoTrackRefiner(network,renderer,plan['objectId'],'cpu')
        current={}
        def pre(_module, inputs):
            current['started']=time.monotonic()
        def post(_module, inputs, outputs):
            elapsed=time.monotonic()-current['started']
            arrays={name: value.detach().cpu().numpy().copy() for name,value in zip(
                ('query','template','rendered_mask','flow','confidence'),(*inputs,*outputs))}
            if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('nonfinite network tensors')
            number=len(current['forwards']);relative=f'{variant}/{current["caseId"]}-forward-{number}.npz'
            p=output/relative;p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**arrays)
            current['forwards'].append(dict(path=relative,sha256=sha(p),networkSeconds=elapsed,
                arrays={k:array_info(a) for k,a in arrays.items()}))
            check_disk(output)
        handles += [network.register_forward_pre_hook(pre),network.register_forward_hook(post)]
        try:
            for case in plan['cases']:
                with Image.open(case['rgbPath']) as im:rgb=np.asarray(im.convert('RGB'))
                with Image.open(case['maskPath']) as im:mask=np.asarray(im)>0
                current.clear();current.update(caseId=case['caseId'],forwards=[])
                begin=time.monotonic();error=None;candidate=None
                try:
                    candidate=refiner.refine(Frame(case['frameId'],rgb,np.asarray(case['K']).reshape(3,3)),
                        mask,np.asarray(case['seedPoseMetres'],dtype=float))
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                rows.append(dict(caseId=case['caseId'],forwards=current['forwards'],
                    refinementSecondsIncludingCapture=time.monotonic()-begin,
                    returnedCandidate=candidate is not None,error=error,
                    poseMetres=candidate.pose.tolist() if candidate is not None else None,
                    correspondences=len(candidate.weights) if candidate is not None else 0))
                print(json.dumps(dict(variant=variant,case=case['caseId'],forwards=len(current['forwards']),
                    candidate=candidate is not None,error=error)),flush=True)
        finally:
            for handle in handles:handle.remove()
            renderer.close();del refiner,renderer,network;gc.collect()
        all_variants[variant]=dict(provenance=provenance,moduleForwardCounts=counts,cases=rows,
            totalSeconds=time.monotonic()-start)
        write(output/f'{variant}.json',all_variants[variant])
    comparisons=[]
    for a,b,c in zip(all_variants['reference_best']['cases'],all_variants['adapted_export']['cases'],
                     all_variants['frozen_baseline']['cases']):
        comparisons.append(compare_case(output,a,b,c))
    result=dict(schema='gotrack_offline_model_diagnostic_result_v1',status='completed',
        evidenceScope=plan['evidenceScope'],cases=comparisons,variants=all_variants,
        referenceExportParity=all(c['referenceExportParity'] for c in comparisons),
        fullNetworkExecuted=all(all(v>0 for v in r['moduleForwardCounts'].values()) for r in all_variants.values()),
        trainingSteps=0,gpuUsed=False,workbenchAdmissionGranted=False,caveats=plan['caveats'])
    write(output/'results.json',result)
    if sha(__file__)!=plan['adapterSHA256'] or wn.source_closure()['closure_sha256']!=plan['sourceClosureSHA256']:
        raise ValueError('diagnostic source changed during inference')


def compare_case(output,reference,export,baseline):
    import numpy as np
    equal=len(reference['forwards'])==len(export['forwards']) and bool(reference['forwards'])
    differences=[]
    for a,b in zip(reference['forwards'],export['forwards']):
        with np.load(Path(output)/a['path'],allow_pickle=False) as x,np.load(Path(output)/b['path'],allow_pickle=False) as y:
            for name in x.files:
                ok=x[name].shape==y[name].shape and x[name].dtype==y[name].dtype
                difference=float(np.max(np.abs(x[name].astype(float)-y[name].astype(float)))) if ok else None
                equal &= ok and difference<=1e-6
                if name in ('query','template','rendered_mask'):
                    equal &= ok and np.array_equal(x[name],y[name])
                differences.append(dict(field=name,maxAbsDifference=difference,shapeDtypeEqual=ok))
    cross={}
    if export['forwards'] and baseline['forwards']:
        with np.load(Path(output)/export['forwards'][0]['path'],allow_pickle=False) as x,np.load(Path(output)/baseline['forwards'][0]['path'],allow_pickle=False) as y:
            cross['firstInputsByteEqual']=all(np.array_equal(x[k],y[k]) for k in ('query','template','rendered_mask'))
            cross['firstFlowMaxAbsDifference']=float(np.max(np.abs(x['flow']-y['flow'])))
            cross['firstConfidenceMaxAbsDifference']=float(np.max(np.abs(x['confidence']-y['confidence'])))
    poses=reference['poseMetres'],export['poseMetres']
    pose_equal=(poses[0] is None and poses[1] is None) or (poses[0] is not None and poses[1] is not None and np.allclose(poses[0],poses[1],atol=1e-6,rtol=0))
    return dict(caseId=reference['caseId'],referenceExportParity=bool(equal and pose_equal),
                referenceExportDifferences=differences,referenceExportPoseEqual=bool(pose_equal),
                baselineVsAdapted=cross)


def supervise(plan,output,max_seconds,max_rss):
    import psutil
    if not 0<max_seconds<=900 or not 0<max_rss<=7*1024**3:
        raise ValueError('supervision limits must be positive and <=900s/7GiB')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);check_disk(output)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1',XFORMERS_DISABLED='1',PYOPENGL_PLATFORM='egl',
             LIBGL_ALWAYS_SOFTWARE='1',VISUALIZEIT_CLOUD='1',PYTHONDONTWRITEBYTECODE='1',
             MESA_SHADER_CACHE_DISABLE='true')
    start=time.monotonic();peak=0;killed=None
    with (output/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,__file__,'worker','--plan',str(Path(plan).resolve()),
            '--output',str(output.resolve())],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                try:
                    process=psutil.Process(child.pid)
                    rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True) if p.is_running())
                except psutil.NoSuchProcess:
                    break
                peak=max(peak,rss)
                if rss>max_rss:killed='process-group RSS limit exceeded'
                if time.monotonic()-start>max_seconds:killed='wall-time limit exceeded'
                try:check_disk(output)
                except ValueError as exc:killed=str(exc)
                if killed:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(.25)
        finally:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
            code=child.wait()
    write(output/'supervision.json',dict(exitCode=code,killedReason=killed,peakSampledRSSBytes=peak,
        elapsedSeconds=time.monotonic()-start,maxSeconds=max_seconds,maxRSSBytes=max_rss,
        rssSamplingSeconds=.25,gpuUsed=False,trainingSteps=0))
    print(json.dumps(read(output/'supervision.json')))
    if code or killed:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--dataset',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    for name in ('supervise','worker'):
        a=s.add_parser(name);a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        if name=='supervise':
            a.add_argument('--max-seconds',type=float,default=900.);a.add_argument('--max-rss',type=int,default=7*1024**3)
    a=s.add_parser('inspect')
    args=p.parse_args()
    if args.command=='prepare':prepare(args.dataset,args.output)
    elif args.command=='worker':worker(args.plan,args.output)
    elif args.command=='supervise':supervise(args.plan,args.output,args.max_seconds,args.max_rss)
    else:
        artifacts,closure=authenticate();print(json.dumps(dict(runtime=runtime(),missingSources=closure['historical_missing'],
            unreviewedSources=closure['unreviewed_replacements'],artifactPaths={k:str(p) for k,(p,_,_) in artifacts.items()}),indent=2))


if __name__=='__main__':main()
