"""Codex assembly controls: entirely manufactured metadata, no dataset or models."""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import struct
import tempfile
import unittest
from unittest import mock

from tools import workbench_hb_scene3_execution_control as ac

ROOT=Path(__file__).resolve().parents[1]
SCRATCH=ROOT/'.cache/workbench-unit-scratch'
def raw(doc):return json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(value).hexdigest()
def tiny_glb():
    document={'asset':{'version':'2.0'},'buffers':[{'byteLength':36}],
              'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':36}],
              'accessors':[{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3','min':[0,0,0],'max':[.5,.5,0]}],
              'meshes':[{'primitives':[{'attributes':{'POSITION':0},'mode':4}]}],
              'nodes':[{'mesh':0}],'scenes':[{'nodes':[0]}],'scene':0}
    text=raw(document);text+=b' '*((-len(text))%4)
    binary=struct.pack('<9f',0,0,0,.5,0,0,0,.5,0)
    return struct.pack('<4sII',b'glTF',2,12+8+len(text)+8+len(binary))+struct.pack('<I4s',len(text),b'JSON')+text+struct.pack('<I4s',len(binary),b'BIN\0')+binary


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(SCRATCH).free<(1<<30)+(12<<20):self.skipTest('real free-space floor')
        self.temp=tempfile.TemporaryDirectory(prefix='hb-assembly-codex-',dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.workspace=self.root/'workspace';self.workspace.mkdir()
        self.scope=self.workspace/'derived/preparation';self.scope.mkdir(parents=True)
        self.out=self.scope/'assembly-v1'
        self.initialize_documents()

    def initialize_documents(self):
        self.docs={};self.items={}
        self.versions={name:digest((ROOT/name).read_bytes()) for name in ac.SOURCE_FILES}
        self.K=[500.,0.,320.,0.,500.,240.,0.,0.,1.]
        self.pose=[[0.,-1.,0.,.125],[1.,0.,0.,-.05],[0.,0.,1.,1.],[0.,0.,0.,1.]]
        self.ordinal=2
        self.label=f'controlled_init:ground_truth:HB:{ac.mi.DATASET_REVISION}:scene000003:obj000004:instance2:s000009'
        # Derivative bytes are opaque, sealed fixture assets; no geometry/video decoding is permitted.
        self.put('glb',tiny_glb(),binary=True)
        self.put('video',b'manufactured sealed video derivative',binary=True)
        self.put('profile',{'schema':ac.pf.SCHEMA_PROFILE,'scene':'000003','frames':{'rows':[{}]*120},'image':{'width':640,'height':480}})
        self.put('camera_k',{'schema':ac.pf.SCHEMA_K,'profile':self.ref('profile'),'K':{'values':self.K},'adjustments':{'origin_shift':None,'crop':None,'rescale':None}})
        self.put('association',{'schema':ac.media.SCHEMA_ASSOCIATION,'scene':'000003','cad':{'object_id':4},'instance_ordinal':None,'profile':self.ref('profile'),'selection':{'frame_id':'s000009','positive_point_xy':[320,340],'roi_xyxy':[285,270,370,440]}})
        self.put('allowlist',{'schema':ac.media.SCHEMA_ALLOWLIST,'label':'TEST_ONLY','model':{'object_id':4},'inputs':{k:self.ref(k) for k in ('profile','camera_k','association')}})
        self.put('model_receipt',{'schema':ac.media.SCHEMA_MODEL,'label':'TEST_ONLY','object_id':4,'allowlist':self.ref('allowlist'),'glb':self.ref('glb'),'geometry':{'triangles':118136,'decimated':False},'portable_targets':{'triangles':'FAIL'}})
        self.put('video_receipt',{'schema':ac.media.SCHEMA_VIDEO,'label':'TEST_ONLY','allowlist':self.ref('allowlist'),'video':{**self.ref('video'),'workspace_relative':self.items['video']['path'].split('workspace/',1)[1]},'decode':{'sequential_frames':120,'end_of_video_after':120,'dimensions':[640,480],'frame_reader_seek_check':{'exact':True,'frame_id':'s000010'}},'clock':{'physical_fps':None,'physical_timestamps':None,'presentation_fps':30},'frames':[{'frame_id':f's{i:06d}'} for i in range(120)]})
        self.put('prerequisites',{'schema':ac.media.SCHEMA_PREREQ,'label':'TEST_ONLY','status':'prepared_media_only','scene':'000003','inputs':{k:self.ref(k) for k in ('allowlist','model_receipt','video_receipt','profile','camera_k','association')},'outputs':{k:self.ref(k) for k in ('glb','video')},'association':{'instance_ordinal':None}})
        self.put('seed_policy',{'schema':ac.sd.SCHEMA_POLICY,'object_id':4,'frame_id':'s000009','association_sha256':self.items['association']['sha256'],'prerequisites_sha256':self.items['prerequisites']['sha256']})
        self.put('seed_plan',{'schema':ac.sd.SCHEMA_PLAN,'label':'TEST_ONLY','policy':self.ref('seed_policy'),'inputs':{'prerequisites':self.ref('prerequisites')},'input_budget':{'total':0}})
        self.put('controlled_seed',{'schema':ac.sd.SCHEMA_SEED,'label':'TEST_ONLY','object_id':4,'frame_id':'s000009','frame_index':9,'instance_ordinal':2,'seed_label':self.label,'seed_pose_m':self.pose,'plan':self.ref('seed_plan'),'applies_once_to_frame_id':'s000010','reuse_on_recovery':False,'other_pose_values_converted':0,'principal_point_origin':None,'K_operation':None,'flags':{'ground_truth_seed':True,'controlled_init':True,'initial_ground_truth_read':True,'automatic_geometry_annotation_access':False,'scored_pose_values_materialized':False}})
        self.put('seed_binding',{'schema':ac.sd.SCHEMA_BINDING,'label':'TEST_ONLY','status':'controlled_seed_extracted','seed':self.ref('controlled_seed'),'plan':self.ref('seed_plan'),'derived_instance_ordinal':2,'seed_label':self.label,'frame9':{'obj_id_4_matches':1,'other_pose_values_converted':0},'association':{'instance_ordinal':None,'unchanged':True}})
        self.put('seed_source_acceptance',{'schema':'workbench_hb_scene3_seed_source_acceptance_v1','verdict':'accepted','independent_review':'accepted','codex_failures':0,'source_pins':self.versions})
        for key,stage in [('actual_seed_plan','plan'),('actual_seed_extraction','extract')]:
            result={'exit':0,'label':'TEST_ONLY','annotation_opens':0 if stage=='plan' else 4}
            if stage=='plan':result['plan_sha256']=self.items['seed_plan']['sha256']
            else:result.update(seed_sha256=self.items['controlled_seed']['sha256'],binding_sha256=self.items['seed_binding']['sha256'],instance_ordinal=2,seed_label=self.label)
            self.put(key,{'schema':'workbench_hb_scene3_actual_seed_execution_v1','stage':stage,'status':'completed','final_closure_returned_successfully':True,'source_acceptance':self.ref('seed_source_acceptance'),'result':result,'neural_executed':False,'registered_or_admitted':False})
        self.put('origin_review',{'schema':'workbench_hb_scene3_principal_point_review_v1','selected_origin':'integer_centers','evidence':'reviewed_assumption','verified_dataset_convention':False,'physical_calibration_proof':False,'origin_shift':[0.,0.],'crop':None,'rescale':None,'raw_K':self.K,'effective_K':self.K,'camera_k':self.ref('camera_k')})
        self.put('origin_independent_review',{'schema':'workbench_hb_scene3_origin_independent_review_v1','verdict':'accepted_as_reviewed_assumption_only','reviewed_sha256':self.items['origin_review']['sha256'],'principal_point_origin':'integer_centers','raw_K_unchanged':True,'origin_shift':[0,0],'verified_dataset_convention':False,'physical_calibration_proof':False,'GT_used':False})
        art={'schema':'workbench_neural_artwork_v1','glb_sha256':self.items['glb']['sha256'],'seed_label':self.label,'anchor_pixel':[320.,340.],'anchor_frame_id':'s000009','effective_K':self.K,'principal_point_origin':'integer_centers','anchor_mask_checked':False,'mesh_faces':118136,'units':'metres','artwork':{'kind':'checkerboard','edge_px':256,'generation':'fixed_builtin_v1','sha256':'a'*64},'marker':{'artwork_sha256':'a'*64}}
        golden_checker='d0c29333966f52c7bcad9f26c1ca17b10696880cd64bfee3790d88c2c8ab970c'
        art['artwork']['sha256']=golden_checker
        art['support_source']='metric CAD geometry (not observation-derived)'
        art['patch_faces']=1
        art['marker']={'face_id':0,'barycentric':[.25,.5,.25],'tangent':[1.,0.,0.],'bitangent':[0.,1.,0.],
                       'normal':[0.,0.,1.],'size_object_units':.025,'artwork_sha256':golden_checker}
        self.put('opening_placement',{'schema':'workbench_hb_scene3_opening_placement_review_v1','verdict':'frozen_opening_ray_hit; observed_mask_review_pending','observed_mask_checked':False,'selection_reviewed':False,'point_changed_or_searched':False,'positive_point_xy':[320,340],'roi_xyxy':[285,270,370,440],'frame_id':'s000009','principal_point_origin':'integer_centers','origin_evidence':'reviewed_assumption','physical_calibration_verified':False,'mesh_triangles':118136,'portable_triangle_target_passed':False,'registered_or_admitted':False,'neural_executed':False,'scored_frame_used':False,'segmentation':False,'annotation_or_raw_camera_opens':0,'texture_edge_px':256,'actual_seed_receipt':self.ref('actual_seed_extraction'),'inputs':{k:self.ref(v) for k,v in [('glb','glb'),('seed','controlled_seed'),('binding','seed_binding'),('K','camera_k'),('association','association'),('origin','origin_review')]},'artwork':art,'artwork_record_sha256':ac.wc.canonical_sha256(art),'depth_m':1.,'projection_self_consistency_residual_px':0.,'face_id':0,'barycentric':[.25,.5,.25],'point_object_m':[.1,.2,.3],'sources':self.versions})
        self.docs['opening_placement']['point_object_m']=[.25,.125,0.]
        self.put('opening_placement',self.docs['opening_placement'])
        self.put('actual_seed_acceptance',{'schema':'workbench_hb_scene3_actual_seed_acceptance_v1','verdict':'accepted','final_closure_accepted':True,'object_id':4,'frame_id':'s000009','instance_ordinal':2,'unique_matching_records':1,'other_pose_values_converted':0,'translation_units':'metres','evidence':{k:self.ref(k) for k in ('actual_seed_extraction','actual_seed_plan','controlled_seed','opening_placement','seed_binding','seed_plan')}})
        self.put('actual_media',{'schema':'workbench_hb_scene3_actual_media_execution_v1','tool_sha256':self.versions['tools/workbench_hb_scene3_media.py'],'neural_executed':False,'gt_or_raw_camera_read':False,'registered_or_admitted':False,'model':{'triangles':118136},'artifacts':{self.items[k]['path']:{p:self.items[k][p] for p in ('bytes','sha256')} for k in ('allowlist','glb','model_receipt','prerequisites','video_receipt','video')}})
        self.registry={'schema':'workbench_hb_scene3_assembly_inputs_v1','timestamp_utc':'2026-10-22T00:00:00Z','reviewer':'Codex manufactured metadata fixture','label':'TEST_ONLY','scene':'000003','object_id':4,'family':'hb_primesense_640x480','head_sha256':'27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311','preparation_scope':self.rel(self.scope),'output_directory':self.rel(self.out),'inputs':self.items,'accepted_source_versions':self.versions,'trust':{'trust_level':'historical_local_integrity','upstream_authenticated':False},'constraints':copy.deepcopy(ac.EXPECTED_CONSTRAINTS)}
        self.registry_path=self.root/'registry.json';self.context=self.seal_registry()

    def rel(self,path):return path.relative_to(self.root).as_posix()
    def ref(self,key):return {p:self.items[key][p] for p in ('path','sha256','bytes')}
    def put(self,key,doc,binary=False):
        kind='geometry_derivative' if key=='glb' else 'video_derivative' if key=='video' else 'metadata'
        parent=self.scope/key if key in ac.IN_SCOPE_KEYS else self.root/'reviews'/key
        if not getattr(self,'memory_only',False):parent.mkdir(parents=True,exist_ok=True)
        path=parent/('asset.bin' if binary else 'record.json')
        data=doc if binary else raw(doc)
        if not getattr(self,'memory_only',False):path.write_bytes(data)
        self.items[key]={'path':self.rel(path),'sha256':digest(data),'bytes':len(data),'cap_bytes':ac.MAX_INPUT_CAPS[kind],'kind':kind}
        if not binary:self.docs[key]=doc
    def seal_registry(self):
        data=raw(self.registry)
        if not getattr(self,'memory_only',False):self.registry_path.write_bytes(data)
        return ac.AssemblyContext(self.root,self.workspace,ac.SyntheticRegistryPin('registry.json',digest(data),len(data)))

    def test_inspection_is_read_only_and_keeps_nonzero_derived_ordinal(self):
        before={self.rel(p):digest(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()}
        with mock.patch.object(ac.sd,'extract_unique_seed',side_effect=AssertionError('seed stage')),mock.patch.object(ac.media,'prepare_model',side_effect=AssertionError('media stage')):
            result=ac.inspect_assembly(self.context)
        self.assertEqual(result['status'],'ready_to_assemble');self.assertEqual(result['derived_instance_ordinal'],2)
        self.assertFalse(result['prefix_attestation']['prefix_bound_attested'])
        self.assertEqual(before,{self.rel(p):digest(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()})

    def test_complete_assembly_round_trip_preserves_pose_clock_masks_and_shared_anchor(self):
        result=ac.assemble(self.context,out=self.out)
        self.assertEqual(set(p.name for p in self.out.iterdir()),{'metric-bundle.json','clip-proposal.json','execution-control-sidecar.json','assembly-receipt.json'})
        bundle=json.loads((self.out/'metric-bundle.json').read_bytes());clip=json.loads((self.out/'clip-proposal.json').read_bytes());side=json.loads((self.out/'execution-control-sidecar.json').read_bytes())
        self.assertEqual(bundle['seed_pose_m'],self.pose);self.assertEqual(bundle['glb_path'],self.items['glb']['path'])
        spec=ac.wc.ClipSpec.from_dict(clip);self.assertEqual(len(spec.frames),120);self.assertEqual(len(spec.scoring_frame_ids),110)
        self.assertTrue(all(f.timestamp_s is None and f.media_time_s is None for f in spec.frames));self.assertFalse(spec.onboarding.selection.reviewed)
        self.assertFalse(side['origin']['physical_calibration_proof']);self.assertTrue(side['shared_anchor'])
        self.assertEqual(side['roles']['primary_neural']['anchor_record_sha256'],side['roles']['frozen_baseline']['anchor_record_sha256'])
        self.assertFalse(side['roles']['primary_neural']['rearm_seed_on_loss']);self.assertEqual(result['exit'],0)
        self.assertEqual(ac.validate_output(self.context,out=self.out)['exit'],0)

    def test_production_cannot_consume_synthetic_registry(self):
        with self.assertRaises(ac.mi.InputsError):ac.inspect_assembly(ac.AssemblyContext(self.root,self.workspace))

    def test_changed_registered_bytes_refuse_before_output(self):
        (self.root/self.items['controlled_seed']['path']).write_bytes(b'changed')
        with self.assertRaises(ac.mi.InputsError):ac.assemble(self.context,out=self.out)
        self.assertFalse(self.out.exists())

    def test_output_collision_preserves_foreign_file(self):
        self.out.mkdir();foreign=self.out/'foreign.json';foreign.write_bytes(b'foreign')
        with self.assertRaises(ac.mi.InputsError):ac.assemble(self.context,out=self.out)
        self.assertEqual(foreign.read_bytes(),b'foreign')

    def test_rehashed_registry_path_escape_and_type_drift_refuse(self):
        original=copy.deepcopy(self.registry)
        for change in [lambda d:d['inputs']['glb'].__setitem__('path','../escape.glb'),lambda d:d['constraints'].__setitem__('annotation_opens',False),lambda d:d.__setitem__('object_id',True)]:
            self.registry=copy.deepcopy(original);change(self.registry)
            with self.subTest(change=change),self.assertRaises(ac.mi.InputsError):ac.inspect_assembly(self.seal_registry())

    def test_rehashed_receipt_cannot_replace_expected_output(self):
        ac.assemble(self.context,out=self.out)
        path=self.out/'metric-bundle.json';doc=json.loads(path.read_bytes());doc['seed_pose_m'][0][3]*=.001;path.write_bytes(raw(doc))
        with self.assertRaises(ac.mi.InputsError):ac.validate_output(self.context,out=self.out)

    def test_missing_final_acceptance_blocks_even_with_seed_and_binding(self):
        (self.root/self.items['actual_seed_acceptance']['path']).unlink()
        with self.assertRaises(ac.mi.InputsError):ac.inspect_assembly(self.context)


class PureAssemblyTests(unittest.TestCase):
    """RAM-only numerical/record controls, usable below the file-producing reserve."""
    def setUp(self):
        self.f=AssemblyTests('runTest');f=self.f;f.memory_only=True
        f.root=ROOT/'.cache/never-written-assembly-fixture';f.workspace=f.root/'workspace'
        f.scope=f.workspace/'derived/preparation';f.out=f.scope/'assembly-v1'
        f.initialize_documents()

    def facts(self):
        return ac._check_relations(self.f.registry,self.f.docs,label='TEST_ONLY',root=self.f.root)

    def relink(self):
        f=self.f
        def replace(value,old,new,newbytes):
            if isinstance(value,dict):
                if value.get('sha256')==old and 'bytes' in value:value['bytes']=newbytes
                for key,item in value.items():value[key]=replace(item,old,new,newbytes)
            elif isinstance(value,list):value[:]=[replace(item,old,new,newbytes) for item in value]
            elif value==old:return new
            return value
        for _ in range(40):
            changed=False
            for key,doc in f.docs.items():
                data=raw(doc);old=f.items[key]['sha256'];new=digest(data)
                if old==new:continue
                changed=True;f.items[key].update(sha256=new,bytes=len(data))
                for target in f.docs.values():replace(target,old,new,len(data))
            if not changed:return
        self.fail('manufactured metadata graph failed to settle')

    def test_manufactured_record_chain_and_nonzero_ordinal(self):
        facts=self.facts();self.assertEqual(facts['ordinal'],2);self.assertEqual(facts['seed_pose'],self.f.pose)

    def test_exact_clip_clock_pose_selection_and_shared_anchor_in_memory(self):
        outputs=ac.build_outputs(self.f.registry,self.facts(),label='TEST_ONLY')
        ac.check_outputs(outputs,root=self.f.root)
        bundle=outputs['metric-bundle.json'];clip=outputs['clip-proposal.json'];side=outputs['execution-control-sidecar.json']
        self.assertEqual(bundle['seed_pose_m'],self.f.pose);self.assertEqual(bundle['glb_path'],self.f.items['glb']['path'])
        self.assertEqual([f['source_frame_index'] for f in clip['frames']],list(range(120)))
        self.assertEqual(clip['scoring_frame_ids'],[f's{i:06d}' for i in range(10,120)])
        self.assertTrue(all(f['timestamp_s'] is None and f['media_time_s'] is None for f in clip['frames']))
        self.assertFalse(clip['onboarding']['selection']['reviewed']);self.assertFalse(side['origin']['physical_calibration_proof'])
        self.assertEqual(side['roles']['primary_neural']['anchor_record_sha256'],side['roles']['frozen_baseline']['anchor_record_sha256'])
        self.assertFalse(side['roles']['primary_neural']['rearm_seed_on_loss'])

    def test_record_contradictions_refuse_after_all_hash_links_updated(self):
        cases=[('seed_binding','derived_instance_ordinal',1),('seed_binding','seed_label','controlled_init:wrong'),
               ('controlled_seed','reuse_on_recovery',True),('actual_seed_extraction','final_closure_returned_successfully',False),
               ('actual_seed_acceptance','final_closure_accepted',False),('actual_seed_acceptance','translation_units','millimetres'),
               ('origin_review','physical_calibration_proof',True),('origin_review','origin_shift',[.5,.5]),
               ('opening_placement','verdict','ray_miss'),('opening_placement','observed_mask_checked',True),
               ('opening_placement','positive_point_xy',[321,340])]
        for key,field,value in cases:
            self.setUp();self.f.docs[key][field]=value;self.relink()
            with self.subTest(key=key,field=field),self.assertRaises(ac.mi.InputsError):self.facts()

    def test_improper_rotation_and_duplicate_object_claim_refuse(self):
        for mutate in [lambda d:d['controlled_seed']['seed_pose_m'][0].__setitem__(0,2.),
                       lambda d:d['seed_binding']['frame9'].__setitem__('obj_id_4_matches',2)]:
            self.setUp();mutate(self.f.docs);self.relink()
            with self.subTest(mutate=mutate),self.assertRaises(ac.mi.InputsError):self.facts()

    def test_boolean_depth_and_negative_residual_are_not_valid_geometry_evidence(self):
        for field,value in [('depth_m',True),('projection_self_consistency_residual_px',-1.)]:
            self.setUp();self.f.docs['opening_placement'][field]=value;self.relink()
            with self.subTest(field=field),self.assertRaises(ac.mi.InputsError):self.facts()

    def test_face_barycentric_and_fixed_checker_semantics(self):
        for field,value in [('face_id',True),('face_id',-1),('face_id',118136),('barycentric',[2.,0.,-1.]),
                            ('projection_self_consistency_residual_px',999.)]:
            self.setUp();self.f.docs['opening_placement'][field]=value;self.relink()
            with self.subTest(field=field,value=value),self.assertRaises(ac.mi.InputsError):self.facts()
        self.setUp();pl=self.f.docs['opening_placement'];pl['artwork']['artwork']['sha256']='b'*64
        pl['artwork']['marker']['artwork_sha256']='b'*64
        pl['artwork_record_sha256']=ac.wc.canonical_sha256(pl['artwork']);self.relink()
        with self.assertRaises(ac.mi.InputsError):self.facts()

    def test_synthetic_standalone_bundle_and_related_outputs_are_labeled(self):
        outputs=ac.build_outputs(self.f.registry,self.facts(),label='TEST_ONLY')
        family=outputs['metric-bundle.json']['family'];self.assertIn('TEST_ONLY',family)
        self.assertEqual(outputs['clip-proposal.json']['provenance']['family'],family)
        self.assertEqual(outputs['execution-control-sidecar.json']['flags']['family'],family)


if __name__=='__main__':unittest.main()
