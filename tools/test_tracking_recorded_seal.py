"""Tiny manufactured metadata tests; no weights, GT, network or actual run."""
import builtins
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from bench import workbench_contract as wc
from tools import tracking_recorded_seal as seal


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(wc.canonical_json(value)+b'\n')


def manufactured(root,obj=7):
    """Full schedule/600 tiny rows, but input audit MUST be explicitly mocked."""
    source=root/'source';workspace=root/'workspace';support=root/'support'
    source.mkdir();workspace.mkdir();support.mkdir();clip_id=f'offline-hb-scene000002-object{obj:06d}-full'
    K=[2.,0.,.5,0.,2.,.5,0.,0.,1.]
    onboarding=dict(keyframe_ids=['f000009'],selection=dict(frame_id='f000009',positive=[[0,0]],negative=[[1,1]],reviewed=True,reviewed_by='manufactured unit fixture'))
    clip=dict(schema='workbench_clip_v1',clip_id=clip_id,title='manufactured parser fixture, not model execution',source_kind='image_sequence',clock_kind='frame_index',
        clock_source='unit frame indices',width=2,height=2,media=dict(root_id='unit',path='rgb',sha256=None,bytes=None),
        frames=[dict(frame_id=f'f{i:06d}',source_frame_index=i,timestamp_s=None,media_time_s=None,image_path=f'{i:06d}.png',image_sha256='e'*64) for i in range(9,310)],
        intrinsics=dict(K=K,status='dataset_reference',source='manufactured calibration'),onboarding=onboarding,scoring_frame_ids=seal.SCHEDULE,
        evidence_role='manufactured_unit_test_fixture',evaluator_inputs_read=False,coordinates=dict(wc.COORDINATE_CONVENTION))
    write(workspace/f'clips/{clip_id}/clip.json',clip)
    write(workspace/f'metric-bundles/{clip_id}.json',dict(note='fixture: no actual CAD; label-free input audit explicitly mocked'))
    manifest=dict(clip_id=clip_id,mask_set='unit-mask',evaluator_inputs_read=False,cancelled=False,
        producer=dict(kind='sam2.1_video_prompted',note='manufactured metadata, no actual SAM'),masks={fid:dict(note='mocked input audit') for fid in seal.SCHEDULE})
    write(workspace/f'masks/{clip_id}/unit-mask/manifest.json',manifest)
    # Unrelated existing cache entries are ALLOWED, not treated as a cache hit.
    old=workspace/'derived/stage-cache/entries/old-object/record.json';write(old,dict(manufactured='prior object cache'))
    pin=dict(schema='tracking_recorded_identity_pin_v1',artifact_id='unit-model',export_sha256='a'*64,export_bytes=1,
        base_checkpoint_sha256='b'*64,installed_head_state_sha256=dict(primary_neural='c'*64,frozen_baseline='d'*64))
    pin_path=root/'pin.json';write(pin_path,pin)
    native=dict(schema='workbench_run_v1',run_id='run-unit',clip_id=clip_id,mode='regression',onboarding=onboarding,
        scheduled_frame_ids=seal.SCHEDULE,roles=seal.ROLES,primary={key:pin[key] for key in ('artifact_id','export_sha256','export_bytes')},
        surface_package_id=None,artwork_kind='checkerboard',config=dict(mask_set='unit-mask'))
    run=workspace/'runs/run-unit';run.mkdir(parents=True)
    write(run/'run.json',dict(run=native,created_at=time.time()+10))
    rows=[]
    for role in seal.ROLES:
        identity=dict(role=role,base_checkpoint_sha256=pin['base_checkpoint_sha256'],installed_head_state_sha256=pin['installed_head_state_sha256'][role],
            installed_head_sha256=pin['export_sha256'] if role=='primary_neural' else None,
            artifact_lineage=dict(mode='workbench_fresh_recorded_artifact_v1',proof_sha256='7'*64,historical_receipts_verified=False,historical_small_records_verified=False),
            admission_contract_sha256='8'*64)
        if role=='primary_neural':identity.update({key:pin[key] for key in ('artifact_id','export_sha256','export_bytes')})
        else:identity['head']='frozen_base_pose_head'
        for i in range(10,310):
            rows.append(wc.FrameResult('run-unit',clip_id,f'f{i:06d}',i,None,None,role,identity,'observed','recovering','suppressed',
                'manufactured temporal-check fixture',None,None,'e'*64,f'f{i:06d}','f'*64,dict(iterations=5)).to_dict())
    (run/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    sources={}
    for name in [*seal.SOURCE_FILES,'tools/tracking_native_recorded.py']:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# manufactured source '+name+'\n')
        if name in seal.SOURCE_FILES:sources[name]=seal.sha(path)
    report=dict(status='completed',run_id='run-unit',clip_id=clip_id,scheduled_frames=300,
        telemetry=dict(run_cache_mode='cold',cache={role:dict(status='miss') for role in seal.ROLES}),
        stages={role:dict(status='completed',frames=300,execution=dict(computed_by='fresh_boundary_execution',
            execution_path='workbench_frame_boundary_v1_main_pipeline',boundary_source_sha256=sources['bench/workbench_frame_boundary.py'])) for role in seal.ROLES},
        dependencies=dict(sources={Path(name).stem:value for name,value in sources.items() if name.endswith('.py')}))
    write(run/'report.json',report)
    arguments=['--workspace',str(workspace),'run',clip_id,'--mode','regression','--roles',','.join(seal.ROLES),'--masks','unit-mask','--device','cpu',
        '--recorded-artifact','operator-proof.json','--admission','operator-admission','--metric-bundle','operator-bundle.json','--head','operator-head.pt']
    freeze=dict(nativeCLIArgs=arguments,budgetProfile='recorded300pair',sourceSHA256=sources,runnerSHA256=seal.sha(source/'tools/tracking_native_recorded.py'),
        maxSeconds=5400,maxRSSBytes=7*1024**3,maxOutputGrowthBytes=512*1024**2,minimumFreeDiskBytes=10*1024**3,trainingSteps=0)
    write(support/'input-freeze.json',freeze)
    write(support/'runtime.json',dict(trainingSteps=0,CUDAAvailable=False,torchThreads=8,torchInteropThreads=1,openCVThreads=1))
    write(support/'supervision.json',dict(exitCode=0,killedReason=None,sourceDrift=[],trainingSteps=0,maxSeconds=5400,maxRSSBytes=7*1024**3,
        elapsedSeconds=1,peakSampledRSSBytes=100,rssSamplingSeconds=.25))
    write(support/'observed-full-forward-counts.json',{f'network_{index}':dict(backbone=1500,decoder=1500,pose_head=1500) for index in range(2)})
    inventory=root/'cache-before.json'
    with patch.object(seal,'ROOT',source):seal.inventory(workspace,clip_id,'unit-mask',inventory)
    return dict(source=source,workspace=workspace,run=run,support=support,inventory=inventory,pin=pin_path,output=root/'prediction-seal.json',clip_id=clip_id)


class SealTests(unittest.TestCase):
    def make(self,fixture,*,audit_result=None):
        # This explicit mock isolates terminal receipt checks from missing tiny
        # RGB/CAD inputs. Actual pure input audit is exercised separately below.
        result=audit_result if audit_result is not None else (None,None,None,None,None,None,[dict(frame_id='unit')],dict(unit_mock=True))
        with patch.object(seal,'ROOT',fixture['source']),patch.object(seal.audit,'verified_inputs',return_value=result) as auditor:
            value=seal.make_seal(fixture['workspace'],fixture['run'],fixture['support'],fixture['inventory'],fixture['output'],
                pin_path=fixture['pin'],pin_sha256=seal.sha(fixture['pin']))
        auditor.assert_called_once();return value

    def test_manufactured_metadata_requires_explicit_input_audit_and_seals_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            f=manufactured(Path(temp));before=seal.sha(f['inventory']);result=self.make(f);body=seal.load(f['output'])
            self.assertEqual(result['rows'],600);self.assertEqual(result['actualFullNetworkForwards'],3000)
            self.assertEqual(body['comparison_id'],'hb-scene000002-object000007-recorded300-v1')
            self.assertEqual(body['scope']['training_overlap'],'not_audited');self.assertTrue(body['prior_reference_access'])
            self.assertFalse(body['inference_process_scored_GT_access']);self.assertNotIn('scoredGroundTruthOpenedBeforeSeal',body)
            self.assertEqual(body['producerSHA256'],seal.sha(seal.__file__));self.assertEqual(seal.sha(f['inventory']),before)
            with self.assertRaisesRegex(ValueError,'already exists'):self.make(f)

    def test_nonempty_inventory_allowed_and_distinct_object_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            f=manufactured(Path(temp),obj=21);inv=seal.load(f['inventory']);self.assertEqual(len(inv['entries']),1)
            self.assertEqual(inv['binding']['clipID'],f['clip_id']);self.make(f)
            self.assertIn('object000021',seal.load(f['output'])['comparison_id'])

    def test_each_terminal_or_cache_failure_prevents_seal(self):
        mutations=[('supervision.json','exitCode',1),('supervision.json','killedReason','platform blocked'),('supervision.json','sourceDrift',['changed']),
            ('input-freeze.json','budgetProfile','pilot'),('runtime.json','CUDAAvailable',True)]
        for filename,key,value in mutations:
            with self.subTest(filename=filename,key=key),tempfile.TemporaryDirectory() as temp:
                f=manufactured(Path(temp));p=f['support']/filename;d=seal.load(p);d[key]=value;write(p,d)
                with self.assertRaises(ValueError):self.make(f)
                self.assertFalse(f['output'].exists())
        for mode in ('hit','off'):
            with tempfile.TemporaryDirectory() as temp:
                f=manufactured(Path(temp));p=f['run']/'report.json';d=seal.load(p)
                if mode=='hit':d['telemetry']['cache']['primary_neural']['status']='hit'
                else:d['telemetry']['run_cache_mode']='off'
                write(p,d)
                with self.assertRaisesRegex(ValueError,'COLD/MISS'):self.make(f)

    def test_incomplete_schedule_or_baseline_only_rejected(self):
        for role_only in (True,False):
            with tempfile.TemporaryDirectory() as temp:
                f=manufactured(Path(temp));p=f['run']/'run.json';d=seal.load(p)
                if role_only:d['run']['roles']=['frozen_baseline']
                else:d['run']['scheduled_frame_ids'].pop()
                write(p,d)
                with self.assertRaises(ValueError):self.make(f)

    def test_changed_source_or_head_or_forward_counts_rejected(self):
        for failure in ('source','head','counts'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temp:
                f=manufactured(Path(temp))
                if failure=='source':(f['source']/'bench/workbench_neural.py').write_text('changed')
                elif failure=='head':
                    p=f['run']/'results.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['backend_identity']['installed_head_state_sha256']='0'*64;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                else:
                    p=f['support']/'observed-full-forward-counts.json';d=seal.load(p);d['network_0']['backbone']=1499;write(p,d)
                with self.assertRaises(ValueError):self.make(f)
                self.assertFalse(f['output'].exists())

    def test_failed_label_free_audit_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            f=manufactured(Path(temp))
            with patch.object(seal,'ROOT',f['source']),patch.object(seal.audit,'verified_inputs',side_effect=ValueError('actual input identity differs')):
                with self.assertRaisesRegex(ValueError,'input identity'):
                    seal.make_seal(f['workspace'],f['run'],f['support'],f['inventory'],f['output'],pin_path=f['pin'],pin_sha256=seal.sha(f['pin']))
            self.assertFalse(f['output'].exists())

    def test_inventory_future_or_different_clip_rejected(self):
        for key,value in [('createdAt',time.time()+10000),('binding',dict(clipID='different'))]:
            with tempfile.TemporaryDirectory() as temp:
                f=manufactured(Path(temp));d=seal.load(f['inventory']);d[key]=value;write(f['inventory'],d)
                with self.assertRaises(ValueError):self.make(f)

    def test_run_and_support_symlinks_and_escaping_inventory_rejected(self):
        for where in ('run','support'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);f=manufactured(root);outside=root/'outside';outside.write_text('must not be read')
                (f[where]/'bad-link').symlink_to(outside)
                with self.assertRaisesRegex(ValueError,'Symlink'):self.make(f)
        with tempfile.TemporaryDirectory() as temp:
            f=manufactured(Path(temp));d=seal.load(f['inventory']);d['entries'][0]['path']='../../escape';write(f['inventory'],d)
            with self.assertRaisesRegex(ValueError,'Escaping'):self.make(f)

    def test_unknown_reference_file_inside_run_never_opened(self):
        with tempfile.TemporaryDirectory() as temp:
            f=manufactured(Path(temp));(f['run']/'evaluator.json').write_text('must not read')
            with patch.object(seal,'sha',wraps=seal.sha) as hashing:
                with self.assertRaisesRegex(ValueError,'Unexpected file'):self.make(f)
                self.assertFalse(any(Path(call.args[0]).name=='evaluator.json' for call in hashing.call_args_list))

    def test_explicit_identity_pin_requires_independent_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            f=manufactured(Path(temp))
            with self.assertRaisesRegex(ValueError,'independent SHA256'):seal.identity_pin(f['pin'])
            with self.assertRaisesRegex(ValueError,'bytes differ'):seal.identity_pin(f['pin'],'0'*64)

    def test_default_identity_is_anchored_to_public_runtime_config(self):
        pin,evidence=seal.identity_pin()
        self.assertEqual(evidence['runtimeConfigSHA256'],seal.PUBLIC_CONFIG_SHA256)
        self.assertEqual(pin['installed_head_state_sha256'],seal.PUBLIC_HEAD_STATES)


class LabelFreeAuditTests(unittest.TestCase):
    def test_real_tiny_input_audit_imports_no_models_and_reads_no_weights_or_gt(self):
        from bench.test_tracking_offline_evaluate import combined_pre_gt_fixture
        original_import=builtins.__import__;original_open=builtins.open;original_os_open=os.open;original_io_open=io.open
        def guarded_import(name,*args,**kwargs):
            if name=='torch' or name.startswith(('torch.','sam2','utils.gotrack','models.')):raise AssertionError('Model runtime import forbidden')
            return original_import(name,*args,**kwargs)
        def check(path):
            if isinstance(path,(str,os.PathLike)):
                p=Path(path)
                if p.suffix in ('.pt','.pth') or p.name in ('evaluator.json','scene_gt.json','scene_gt_info.json') or 'mask_visib' in p.parts:
                    raise AssertionError('Scored reference/weight access forbidden')
        def guarded_open(path,*args,**kwargs):check(path);return original_open(path,*args,**kwargs)
        def guarded_io_open(path,*args,**kwargs):check(path);return original_io_open(path,*args,**kwargs)
        def guarded_os_open(path,*args,**kwargs):check(path);return original_os_open(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp,patch('builtins.__import__',side_effect=guarded_import),patch('builtins.open',side_effect=guarded_open),patch('io.open',side_effect=guarded_io_open),patch('os.open',side_effect=guarded_os_open):
            root=Path(temp);workspace,run,sp,sha=combined_pre_gt_fixture(root)
            s,native,rows=seal.audit.verify_seal(run,sp,sha,True)
            seal.audit.verify_comparison_seals(s,[])
            out=seal.audit.verified_inputs(workspace,run,native,s,rows)
            self.assertEqual(len(out[6]),1);self.assertEqual(out[7]['seedUsageByRole']['primary_neural']['seed_uses'],0)


if __name__=='__main__':unittest.main()
