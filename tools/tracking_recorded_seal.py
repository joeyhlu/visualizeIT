"""Inventory then seal a completed native combined HB300 recorded run.

Label-free metadata/file audit only: no evaluator labels, model construction,
Torch/SAM/GoTrack network imports, inference, admission, or receipt fabrication.
Raw operator receipts remain local. A seal is not an accuracy/admission claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bench import workbench_contract as wc
from bench import tracking_offline_evaluate as audit

ROLES=['primary_neural','frozen_baseline']
SCHEDULE=[f'f{index:06d}' for index in range(10,310)]
CACHE_POLICY='production_enabled_but_cold_first_run'
SOURCE_FILES=(
    'bench/workbench_neural.py','bench/workbench_cli.py','bench/workbench_contract.py',
    'bench/quality_gotrack.py','bench/quality_gotrack_inference.py','bench/workbench_geometry.py',
    'bench/workbench_render.py','bench/workbench_frame_boundary.py','bench/workbench_telemetry.py',
    'bench/runtime/requirements-tracking-sam-cpu.txt')
SUPPORT_FILES=('input-freeze.json','runtime.json','supervision.json','observed-full-forward-counts.json')
PUBLIC_CONFIG='bench/runtime/gotrack-flow-terminal-20261021.json'
PUBLIC_CONFIG_SHA256='05fe3012a18c2ed2669bcf2690e0b48650d9472fc1bffdf0980b1a253e6fc770'
# Already public in29667146 docs/benchmark/tracking-neural-cpu-v1/summary.json,
# whose SHA256 is93ebbe15...; these are measured tensor-state hashes, not files.
PUBLIC_STATE_EVIDENCE_SHA256='93ebbe15f072bea840f4adcfd5ea86ad40a5c0eb8154c48bdaade1d488f1ed40'
PUBLIC_HEAD_STATES={
    'primary_neural':'a3c1e45a61f7819aa841ad0ae3d41c70945f1c360f5ff9c072d3da87c1966dd8',
    'frozen_baseline':'c9e77350048943092f68b01a5439c526453da20e85cef23ed3fb85b09803efd6'}
CLIP_RE=re.compile(r'offline-hb-scene(\d{6})-object(\d{6})-full\Z')
SHA_RE=re.compile(r'[0-9a-f]{64}\Z')


def require(condition,message):
    if not condition:raise ValueError(message)


def safe_path(path,*,directory=False,allow_missing=False):
    """Reject symlinks in every component, including explicit root arguments."""
    path=Path(os.path.abspath(path))
    for part in reversed([path,*path.parents]):
        if part.is_symlink():raise ValueError('Symlink path is forbidden')
    if not path.exists():
        if allow_missing:return path
        raise ValueError('Required path is missing')
    mode=path.stat().st_mode
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode),'Expected regular directory/file')
    return path


def under(root,relative,*,directory=False,allow_missing=False):
    root=safe_path(root,directory=True);relative=Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts and relative.parts,'Escaping/empty relative path')
    path=safe_path(root/relative,directory=directory,allow_missing=allow_missing)
    require(path.is_relative_to(root),'Path escapes declared root')
    return path


def sha(path):
    path=safe_path(path);h=hashlib.sha256()
    flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)
    with os.fdopen(os.open(path,flags),'rb') as stream:
        before=os.fstat(stream.fileno())
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
        after=os.fstat(stream.fileno())
    require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'File changed while hashing')
    return h.hexdigest()


def load(path):
    path=safe_path(path);require(path.stat().st_size<=32*1024**2,'JSON metadata exceeds32MiB')
    with os.fdopen(os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)),'rb') as stream:return wc.loads_strict(stream.read())


def spec(path):return dict(sha256=sha(path),bytes=safe_path(path).stat().st_size)


def tree_files(root):
    root=safe_path(root,directory=True);out=[]
    for parent,dirs,files in os.walk(root,followlinks=False):
        for name in dirs:safe_path(Path(parent)/name,directory=True)
        for name in files:out.append(safe_path(Path(parent)/name))
        require(len(out)<=20000,'Evidence/cache inventory exceeds20000 files')
    return sorted(out)


def immutable_json(output,value):
    output=safe_path(output,allow_missing=True)
    require(not output.exists(),'Immutable output already exists')
    output.parent.mkdir(parents=True,exist_ok=True);safe_path(output.parent,directory=True)
    raw=wc.canonical_json(value)+b'\n'
    with output.open('xb') as stream:stream.write(raw)
    return dict(sha256=sha(output),bytes=len(raw))


def workspace_binding(workspace,clip_id,mask_set):
    workspace=safe_path(workspace,directory=True);match=CLIP_RE.fullmatch(clip_id)
    require(match is not None,'Expected a registered full HB clip ID')
    wc._ident(mask_set,'mask set')
    clip_path=under(workspace,f'clips/{clip_id}/clip.json');clip=load(clip_path);wc.ClipSpec.from_dict(clip)
    manifest_path=under(workspace,f'masks/{clip_id}/{mask_set}/manifest.json');manifest=load(manifest_path)
    require(manifest.get('evaluator_inputs_read') is False and manifest.get('cancelled') is False,'Observed-mask manifest incomplete/label-fed')
    require(manifest.get('clip_id')==clip_id and manifest.get('mask_set')==mask_set,'Observed-mask identity differs')
    require(manifest.get('producer',{}).get('kind')=='sam2.1_video_prompted','Actual prompted SAM producer required')
    require(set(SCHEDULE)<=set(manifest['masks']),'Missing full300 SAM mask records')
    bundle_path=under(workspace,f'metric-bundles/{clip_id}.json')
    return dict(workspaceIdentitySHA256=hashlib.sha256(str(workspace).encode()).hexdigest(),clipID=clip_id,
        clipSHA256=sha(clip_path),maskSet=mask_set,nativeMaskManifestSHA256=wc.canonical_sha256(manifest),
        metricBundleSHA256=sha(bundle_path)),clip,manifest,match


def inventory(workspace,clip_id,mask_set,output):
    binding,_,_,_=workspace_binding(workspace,clip_id,mask_set)
    workspace=safe_path(workspace,directory=True);cache=under(workspace,'derived/stage-cache',directory=True,allow_missing=True)
    output=safe_path(output,allow_missing=True);require(not output.is_relative_to(cache),'Inventory output must be outside cache')
    files=tree_files(cache) if cache.exists() else []
    body=dict(schema='tracking_recorded_cache_inventory_v1',producerSHA256=sha(__file__),cachePolicy=CACHE_POLICY,intendedRoles=ROLES,
        createdAt=time.time(),binding=binding,cacheRootRelative='derived/stage-cache',existed=cache.exists(),
        entries=[dict(path=str(path.relative_to(cache)),**spec(path)) for path in files],
        note='State inventory, not cache-off/global-empty proof. Prior unrelated entries are allowed; terminal native MISS/cold and genuine observed forward counts remain mandatory.')
    result=immutable_json(output,body)
    return dict(schema=body['schema'],files=len(files),existed=cache.exists(),**result)


def identity_pin(path=None,expected_sha=None):
    if path is None:
        require(expected_sha is None,'Identity pin SHA requires an explicit file')
        config_path=under(ROOT,PUBLIC_CONFIG);require(sha(config_path)==PUBLIC_CONFIG_SHA256,'Public identity config bytes differ')
        config=load(config_path)
        pin={key:config[key] for key in ('artifact_id','export_sha256','export_bytes','base_checkpoint_sha256')}
        pin['installed_head_state_sha256']=PUBLIC_HEAD_STATES
        evidence=dict(kind='public296_runtime_and_neural_diagnostic',runtimeConfigSHA256=PUBLIC_CONFIG_SHA256,
            publicHeadStateEvidenceSHA256=PUBLIC_STATE_EVIDENCE_SHA256)
    else:
        require(isinstance(expected_sha,str) and SHA_RE.fullmatch(expected_sha),'Explicit identity pin requires independent SHA256')
        require(sha(path)==expected_sha,'Explicit identity pin bytes differ');pin=load(path)
        require(pin.pop('schema',None)=='tracking_recorded_identity_pin_v1','Explicit identity pin schema differs')
        evidence=dict(kind='explicit_operator_identity_pin',sha256=expected_sha)
    require(set(pin)=={'artifact_id','export_sha256','export_bytes','base_checkpoint_sha256','installed_head_state_sha256'},'Identity pin fields differ')
    wc._ident(pin['artifact_id'],'artifact ID');require(type(pin['export_bytes']) is int and pin['export_bytes']>0,'Export size pin invalid')
    require(all(isinstance(pin[key],str) and SHA_RE.fullmatch(pin[key]) for key in ('export_sha256','base_checkpoint_sha256')),'Weight identity pins invalid')
    require(isinstance(pin['installed_head_state_sha256'],dict) and set(pin['installed_head_state_sha256'])==set(ROLES) and all(isinstance(v,str) and SHA_RE.fullmatch(v) for v in pin['installed_head_state_sha256'].values()),'Head-state pins invalid')
    return pin,evidence


def command_value(arguments,flag):
    require(isinstance(arguments,list) and all(isinstance(v,str) for v in arguments),'Missing native command freeze')
    require(arguments.count(flag)==1,'Missing/duplicate native command option: '+flag)
    index=arguments.index(flag);require(index+1<len(arguments),'Missing native option value')
    return arguments[index+1]


def terminal_evidence(stage_record,workspace,native,report):
    support=safe_path(stage_record,directory=True);tree_files(support) # reject any symlink/special support entry
    paths={name:under(support,name) for name in SUPPORT_FILES};records={name:load(path) for name,path in paths.items()}
    supervision=records['supervision.json'];freeze=records['input-freeze.json'];runtime=records['runtime.json']
    require(type(supervision.get('exitCode')) is int and supervision['exitCode']==0 and supervision.get('killedReason') is None and supervision.get('sourceDrift')==[],'Supervisor did not succeed without guard/source drift')
    require(all(record.get('trainingSteps')==0 for record in (supervision,freeze,runtime)),'Inference-only zero training steps required')
    require(runtime.get('CUDAAvailable') is False and runtime.get('torchThreads')==8 and runtime.get('torchInteropThreads')==1 and runtime.get('openCVThreads')==1,'Measured CPU thread/runtime profile differs')
    require(freeze.get('budgetProfile')=='recorded300pair','Normal combined recorded300pair profile required')
    maximum=freeze.get('maxSeconds');rss=freeze.get('maxRSSBytes')
    require(isinstance(maximum,(int,float)) and 0<maximum<=5400 and isinstance(rss,int) and 0<rss<=7*1024**3,'Resource ceilings exceed bounded profile')
    require(freeze.get('maxOutputGrowthBytes',math.inf)<=512*1024**2 and freeze.get('minimumFreeDiskBytes',0)>=10*1024**3,'Output/disk guards differ')
    require(supervision.get('maxSeconds')==maximum and supervision.get('maxRSSBytes')==rss,'Supervisor/freeze ceilings differ')
    require(0<=supervision.get('elapsedSeconds',math.inf)<=maximum and 0<=supervision.get('peakSampledRSSBytes',math.inf)<=rss and 0<supervision.get('rssSamplingSeconds',math.inf)<=.25,'Terminal runtime measurements violate guards')
    arguments=freeze.get('nativeCLIArgs')
    require(safe_path(command_value(arguments,'--workspace'),directory=True)==workspace,'Frozen command workspace differs')
    require(arguments.count('run')==1 and arguments[arguments.index('run')+1]==native['clip_id'],'Frozen command clip/stage differs')
    for flag,value in (('--roles',','.join(ROLES)),('--mode','regression'),('--masks',native['config']['mask_set']),('--device','cpu')):
        require(command_value(arguments,flag)==value,'Frozen native command differs: '+flag)
    command_value(arguments,'--recorded-artifact');command_value(arguments,'--admission');command_value(arguments,'--metric-bundle');command_value(arguments,'--head')
    require(freeze.get('runnerSHA256')==sha(under(ROOT,'tools/tracking_native_recorded.py')),'Actual supervisor source differs from frozen runner')
    sources=freeze.get('sourceSHA256');require(isinstance(sources,dict) and set(SOURCE_FILES)<=set(sources),'Incomplete inference source freeze')
    for relative,expected in sources.items():
        require(isinstance(expected,str) and SHA_RE.fullmatch(expected),'Invalid source digest')
        require(sha(under(ROOT,relative))==expected,'Source bytes changed since inference: '+relative)
        if relative.startswith('bench/') and relative.endswith('.py'):
            key=Path(relative).stem
            if key in report['dependencies']['sources']:require(report['dependencies']['sources'][key]==expected,'Native report/source freeze differs')
    return records,{name:spec(path) for name,path in paths.items()}


def native_run_files(run):
    files=tree_files(run);out={};require(sum(path.stat().st_size for path in files)<=512*1024**2,'Native evidence exceeds512MiB growth guard')
    for path in files:
        rel=path.relative_to(run);name=rel.as_posix()
        allowed=name in ('run.json','report.json','results.jsonl','neural-artwork.json') or (len(rel.parts)==2 and rel.parts[0]=='stages' and path.suffix=='.json') or (len(rel.parts)==3 and rel.parts[0]=='evidence' and rel.parts[1] in ROLES and re.fullmatch(r'f\d{6}\.png',rel.parts[2]))
        require(allowed,'Unexpected file in native run; no reference/extra file is read')
        out[name]=spec(path)
    require({'run.json','report.json','results.jsonl'}<=set(out),'Native run files incomplete')
    return out


def make_seal(workspace,run,stage_record,cache_inventory,output,*,pin_path=None,pin_sha256=None):
    workspace=safe_path(workspace,directory=True);run=safe_path(run,directory=True);output=safe_path(output,allow_missing=True)
    require(not output.exists(),'Immutable prediction seal already exists')
    require(not output.is_relative_to(run) and not output.is_relative_to(safe_path(stage_record,directory=True)),'Seal must be outside immutable run/support trees')
    before=native_run_files(run);native_record=load(under(run,'run.json'));native=wc.RunSpec.from_dict(native_record['run']).to_dict()
    require(native['roles']==ROLES and native['scheduled_frame_ids']==SCHEDULE and native['mode']=='regression','Expected normal combined full300 native schedule/roles')
    require(run.name==native['run_id'] and run.parent==under(workspace,'runs',directory=True),'Run directory is not bound to native workspace/run ID')
    binding,clip,manifest,match=workspace_binding(workspace,native['clip_id'],native['config']['mask_set'])
    require(clip['scoring_frame_ids']==SCHEDULE and [f['source_frame_index'] for f in clip['frames']]==list(range(9,310)),'Registered full clip differs from source9..309 protocol')
    rows=[wc.FrameResult.from_dict(wc.loads_strict(line)).to_dict() for line in under(run,'results.jsonl').read_text().splitlines() if line]
    require(len(rows)==600 and {(r['backend_role'],r['frame_id']) for r in rows}=={(role,fid) for role in ROLES for fid in SCHEDULE},'Incomplete/duplicate combined prediction rows')
    require(all(r['run_id']==native['run_id'] and r['clip_id']==native['clip_id'] for r in rows),'Prediction run/clip identity differs')
    report=load(under(run,'report.json'))
    require(report.get('status')=='completed' and report.get('run_id')==native['run_id'] and report.get('clip_id')==native['clip_id'] and report.get('scheduled_frames')==300,'Native run is not successfully completed/bound')
    telemetry=report['telemetry'];require(telemetry.get('run_cache_mode')=='cold' and all(telemetry.get('cache',{}).get(role,{}).get('status')=='miss' for role in ROLES),'Expected ENABLED-BUT-COLD/MISS, never cached inference')
    records,support_specs=terminal_evidence(stage_record,workspace,native,report);freeze=records['input-freeze.json']
    for role in ROLES:
        stage=report['stages'][role];execution=stage.get('execution',{})
        require(stage.get('status')=='completed' and stage.get('frames')==300 and execution.get('computed_by')=='fresh_boundary_execution' and execution.get('execution_path')=='workbench_frame_boundary_v1_main_pipeline','Role is not a terminal fresh native boundary execution')
        require(execution.get('boundary_source_sha256')==freeze['sourceSHA256']['bench/workbench_frame_boundary.py'],'Boundary source identity differs')
    pin,pin_evidence=identity_pin(pin_path,pin_sha256)
    require(native['primary']=={key:pin[key] for key in ('artifact_id','export_sha256','export_bytes')},'Native primary identity differs from explicit/public pin')
    lineage_bindings=set()
    for row in rows:
        ident=row['backend_identity'];role=row['backend_role']
        require(ident.get('installed_head_state_sha256')==pin['installed_head_state_sha256'][role] and ident.get('base_checkpoint_sha256')==pin['base_checkpoint_sha256'],'Actual installed head/base identity differs')
        if role=='primary_neural':require(ident.get('artifact_id')==pin['artifact_id'] and ident.get('export_sha256')==pin['export_sha256'] and ident.get('export_bytes')==pin['export_bytes'] and ident.get('installed_head_sha256')==pin['export_sha256'],'Actual adapted export identity differs')
        else:require(ident.get('installed_head_sha256') is None and ident.get('head')=='frozen_base_pose_head','Frozen comparator is not the actual frozen head')
        lineage=ident.get('artifact_lineage',{})
        require(lineage.get('mode')=='workbench_fresh_recorded_artifact_v1' and lineage.get('historical_receipts_verified') is False and lineage.get('historical_small_records_verified') is False,'Fresh recorded lineage is absent/misrepresented')
        require(isinstance(lineage.get('proof_sha256'),str) and SHA_RE.fullmatch(lineage['proof_sha256']) and isinstance(ident.get('admission_contract_sha256'),str) and SHA_RE.fullmatch(ident['admission_contract_sha256']),'Missing actual native proof/admission bindings')
        lineage_bindings.add((lineage['proof_sha256'],ident['admission_contract_sha256']))
    require(len(lineage_bindings)==1,'Roles/frames do not share the same normal proof/admission lineage')
    counts=records['observed-full-forward-counts.json'];support=dict(pose={'observed-full-forward-counts.json':dict(record=counts)})
    forward_audit=audit.verify_forward_counts(rows,ROLES,support)
    cache_path=safe_path(cache_inventory);cache=load(cache_path)
    require(cache.get('schema')=='tracking_recorded_cache_inventory_v1' and cache.get('intendedRoles')==ROLES and cache.get('cachePolicy')==CACHE_POLICY and cache.get('binding')==binding,'Pre-run inventory is not bound to this workspace/clip/masks/bundle')
    require(cache.get('producerSHA256')==sha(__file__),'Pre-run inventory producer differs')
    require(isinstance(cache.get('entries'),list) and len(cache['entries'])<=20000,'Cache inventory entries malformed')
    for entry in cache['entries']:
        relative=Path(entry.get('path',''))
        require(relative.parts and not relative.is_absolute() and '..' not in relative.parts and type(entry.get('bytes')) is int and entry['bytes']>=0 and isinstance(entry.get('sha256'),str) and SHA_RE.fullmatch(entry['sha256']),'Escaping/malformed cache inventory entry')
    require(isinstance(cache.get('createdAt'),(int,float)) and math.isfinite(cache['createdAt']) and cache['createdAt']<=native_record.get('created_at',-math.inf),'Inventory is not declared before native run creation')
    body=dict(schema='offline_recorded_prediction_seal_v2',evaluationProtocol='expanded_development_reused_reference',prior_reference_access=True,
        inference_process_scored_GT_access=False,scored_GT_feedback_to_inference=False,weights_math_thresholds_pinned=True,
        comparative_scoring_requires_all_role_seals=True,expectedComparativeRoles=ROLES,evaluated=False,
        comparison_id=f'hb-scene{match[1]}-object{match[2]}-recorded300-v1',run_id=native['run_id'],scoredFrameIDs=SCHEDULE,rows=600,roles=ROLES,
        files=before,nativeMaskManifestSHA256=binding['nativeMaskManifestSHA256'],supervisionSHA256=support_specs['supervision.json']['sha256'],
        sourceFreezeSHA256=support_specs['input-freeze.json']['sha256'],stageRecordFiles=support_specs,preRunCacheInventorySHA256=sha(cache_path),
        producerSHA256=sha(__file__),inputAuditorSHA256=sha(audit.__file__),identityPin=pin_evidence,cachePolicy=CACHE_POLICY,actualForwardAudit=forward_audit,
        scope=dict(scene_id=match[1],object_id=int(match[2]),profile='full',scored_frames_per_role=300,
            training_overlap='documented_object_overlap' if int(match[2])==20 else 'not_audited',
            reference_use='expanded development/reused reference, never unseen-object/unopened-reference holdout'),
        model_input_scope='native RGB/calibration/metric CAD/actual SAM; only source9 reviewed prompt and known pose; no scored labels')
    # Reviewed LABEL-FREE auditor only; it never loads evaluator.json or models.
    _,_,_,_,_,_,inputs,inference_bindings=audit.verified_inputs(workspace,run,native,body,rows)
    body['verifiedInferenceInputBindings']=inference_bindings;body['nativeInputIdentityIndexSHA256']=wc.canonical_sha256(inputs)
    require(native_run_files(run)==before,'Native run evidence changed during sealing')
    require({name:spec(under(stage_record,name)) for name in SUPPORT_FILES}==support_specs,'Supervisor evidence changed during sealing')
    result=immutable_json(output,body)
    return dict(run_id=native['run_id'],rows=600,roles=ROLES,comparison_id=body['comparison_id'],actualFullNetworkForwards=forward_audit['total_full_network_forwards'],**result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    inv=commands.add_parser('inventory',help='Record existing cache state before run; never delete/require global emptiness')
    for name in ('workspace','output'):inv.add_argument('--'+name,type=Path,required=True)
    inv.add_argument('--clip-id',required=True);inv.add_argument('--mask-set',required=True)
    seal=commands.add_parser('seal',help='Audit a terminal normal combined run and write a new immutable v2 seal')
    for name in ('workspace','run','stage-record','cache-inventory','output'):seal.add_argument('--'+name,type=Path,required=True)
    seal.add_argument('--identity-pin',type=Path);seal.add_argument('--identity-pin-sha256')
    args=parser.parse_args(argv)
    if args.command=='inventory':result=inventory(args.workspace,args.clip_id,args.mask_set,args.output)
    else:result=make_seal(args.workspace,args.run,args.stage_record,args.cache_inventory,args.output,pin_path=args.identity_pin,pin_sha256=args.identity_pin_sha256)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
