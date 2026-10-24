"""Codex manufactured controls for replay evaluation; no actual reference labels/models."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tools import workbench_mask_replay_evaluate as ev

K=np.array([[1000.,0,320.],[0,1000.,240.],[0,0,1.]])
def prediction(dx=.002,**changes):
 T=np.eye(4);T[:3,3]=[dx,0,1]
 r={'pose_state':'tracking','pose_units':'metric','pose':T.tolist(),'render_state':'rendered','evidence':{'render_stats':{'rendered_pixels':100}}};r.update(changes);return r

def fixture():
 refs={f:{'reference_status':'eligible','reference_uv':[320.,240.],'visibility':'visible'} for f in ev.SCHEDULE}
 rows={role:{} for role in ev.ROLES};summaries={}
 for role in ev.ROLES:
  expected=[]
  for i,f in enumerate(ev.SCHEDULE):
   r=prediction();r.update(backend_role=role,frame_id=f)
   if i==0:r.update(pose_state='recovering',pose=None,render_state='suppressed',evidence={})
   rows[role][f]=r
   expected.append({'frame_id':f,'reference_status':'eligible','reference_uv':[320.,240.],'reference_visible':'visible','prediction_status':'no_tracking_pose' if i==0 else 'ok','prediction_uv':None if i==0 else [322.,240.],'error_native_px':None if i==0 else 2.,'error_px720':None if i==0 else 3.})
  metric={'reference_eligible':110,'errors_measured':109,'median_px720':3.,'p95_px720':3.,'coverage_le_5px720':109/110,'coverage_le_10px720':109/110}
  summaries[role]={'A_anchor_rows':expected,'A_anchor':metric,'C_anchor_reference_visible':copy.deepcopy(metric),'A_prediction_status_counts':{'no_tracking_pose':1,'ok':109},'A_reference_status_counts':{'eligible':110},'D_availability':{'frames':110,'tracking_pose':109,'render_state_rendered':109,'rendered_pixels_positive':109,'rendered_pixels_median_when_positive':100.}}
 sealed={'objects':{'2':{'anchor_visibility':[{'frame_id':f,'status':'visible','reason':None} for f in ev.SCHEDULE],'roles':summaries}}}
 plan={'objects':{'2':{'frames':[{'frame_id':f} for f in ev.SCHEDULE]}}}
 return refs,rows,sealed,plan

class ProjectionTests(unittest.TestCase):
 def proj(self,r,anchor=(0,0,0),cam=K):return ev.project_prediction(anchor,cam,r,[640,480])
 def test_identity_and_2mm_at_one_metre(self):
  self.assertEqual(self.proj(prediction(0))['prediction_uv'],[320.,240.])
  pred=self.proj(prediction());self.assertEqual(pred['prediction_uv'],[322.,240.])
  scored=ev.score_anchor({'reference_status':'eligible','reference_uv':[320.,240.],'visibility':'visible'},pred,480)
  self.assertEqual(scored['error_native_px'],2);self.assertEqual(scored['error_px720'],3)
 def test_nonzero_skew_and_metric_translation(self):
  cam=K.copy();cam[0,1]=10;cam[1,1]=900
  self.assertEqual(self.proj(prediction(0),(.01,.02,0),cam)['prediction_uv'],[330.2,258.])
 def test_integer_center_half_pixel_bounds(self):
  for u,status in [(-.5,'ok'),(-.5001,'prediction_out_of_image'),(639.4999,'ok'),(639.5,'prediction_out_of_image')]:
   with self.subTest(u=u):self.assertEqual(self.proj(prediction((u-320)/1000))['prediction_status'],status)
 def test_recovering_pose_is_not_stale_tracking(self):
  self.assertEqual(self.proj(prediction(pose_state='recovering'))['prediction_status'],'no_tracking_pose')
  self.assertEqual(self.proj(prediction(pose=None))['prediction_status'],'no_tracking_pose')
 def test_reflection_wrong_units_nonfinite_and_bad_homogeneous_row(self):
  cases=[prediction(pose_units='relative')]
  for mut in ['reflection','nan','lastrow','shape']:
   r=prediction();T=np.array(r['pose'])
   if mut=='reflection':T[0,0]=-1
   if mut=='nan':T[0,3]=float('nan')
   if mut=='lastrow':T[3,0]=1e-12
   if mut=='shape':T=T[:3]
   r['pose']=T.tolist();cases.append(r)
  for r in cases:self.assertEqual(self.proj(r)['prediction_status'],'invalid_prediction_pose')
 def test_malformed_nonnumeric_pose_is_rejected_as_invalid_row(self):
  r=prediction(pose=[['broken']]*4)
  self.assertEqual(self.proj(r)['prediction_status'],'invalid_prediction_pose')
 def test_behind_camera_and_outside(self):
  r=prediction();r['pose'][2][3]=-1
  self.assertEqual(self.proj(r)['prediction_status'],'prediction_behind_camera')
  self.assertEqual(self.proj(prediction(100))['prediction_status'],'prediction_out_of_image')

class AccountingTests(unittest.TestCase):
 def test_failures_remain_in_reference_denominator(self):
  data=[{'frame_id':str(i),'reference_status':'eligible','prediction_status':'ok' if e is not None else 'no_tracking_pose','error_px720':e} for i,e in enumerate([0.,6.,None,None])]
  s=ev.summarize_anchor(data,['0','1','2','3'])
  self.assertEqual(s['denominator'],4);self.assertEqual(s['measured'],2);self.assertEqual(s['coverage_le_5px720'],.25);self.assertEqual(s['coverage_le_10px720'],.5);self.assertEqual(s['conditional_median_px720'],3.)
 def test_empty_denominator_and_no_predictions_are_distinct(self):
  empty=ev.summarize_anchor([],[]);self.assertIsNone(empty['coverage_le_5px720']);self.assertIsNone(empty['conditional_median_px720'])
  s=ev.summarize_anchor([{'frame_id':'a','reference_status':'eligible','prediction_status':'no_tracking_pose','error_px720':None}],['a']);self.assertEqual(s['denominator'],1);self.assertEqual(s['coverage_le_5px720'],0);self.assertIsNone(s['conditional_median_px720'])
 def test_legacy_pose_presence_separate_from_validated_tracking(self):
  s=ev.summarize_availability([prediction(pose_state='recovering'),prediction()]);self.assertEqual(s['legacy_pose_present'],2);self.assertEqual(s['validated_tracking_poses'],1)
 def test_exact_schedule_and_role_join_missing_duplicate_extra(self):
  refs,rows,_,_=fixture();raw=[r for role in ev.ROLES for r in rows[role].values()];ev.validate_schedule(raw,ev.SCHEDULE)
  for broken in [raw[:-1],raw+[raw[-1]],raw+[dict(raw[-1],frame_id='s000999')],raw+[dict(raw[-1],backend_role='unknown')]]:
   with self.assertRaises(ev.ReplayError):ev.validate_schedule(broken,ev.SCHEDULE)
 def test_reference_cross_role_pixels_and_visibility_cannot_disagree(self):
  _,_,sealed,plan=fixture();self.assertEqual(len(ev.extract_reference(sealed,plan)),110)
  for field,value in [('reference_uv',[321.,240.]),('reference_visible','unknown'),('reference_status','invalid_reference_pose')]:
   bad=copy.deepcopy(sealed);bad['objects']['2']['roles'][ev.ROLES[0]]['A_anchor_rows'][0][field]=value
   with self.assertRaises(ev.ReplayError):ev.extract_reference(bad,plan)
 def test_reference_missing_duplicate_and_visible_ineligible_rejected(self):
  _,_,sealed,plan=fixture()
  bad=copy.deepcopy(sealed);bad['objects']['2']['anchor_visibility'].pop()
  with self.assertRaises(ev.ReplayError):ev.extract_reference(bad,plan)
  bad=copy.deepcopy(sealed);bad['objects']['2']['anchor_visibility'].append(copy.deepcopy(bad['objects']['2']['anchor_visibility'][0]))
  with self.assertRaises(ev.ReplayError):ev.extract_reference(bad,plan)
  bad=copy.deepcopy(sealed)
  for role in ev.ROLES:bad['objects']['2']['roles'][role]['A_anchor_rows'][0]['reference_status']='invalid_reference_pose'
  with self.assertRaises(ev.ReplayError):ev.extract_reference(bad,plan)
 def test_all_220_original_rows_and_summaries_reproduce_analytic_control(self):
  refs,rows,sealed,_=fixture();r=ev.verify_original_control(rows,refs,sealed,anchor_m=[0,0,0],K_by_frame={f:K for f in ev.SCHEDULE},image_size=[640,480]);self.assertTrue(r['passed'],r['mismatches'])
 def test_original_pixel_error_or_summary_mismatch_fails_control(self):
  refs,rows,sealed,_=fixture()
  for mode in ['pixel','error','summary']:
   bad=copy.deepcopy(sealed);s=bad['objects']['2']['roles'][ev.ROLES[0]]
   if mode=='pixel':s['A_anchor_rows'][1]['prediction_uv'][0]+=.01
   if mode=='error':s['A_anchor_rows'][1]['error_px720']=5.00000001
   if mode=='summary':s['A_anchor']['coverage_le_5px720']=1
   result=ev.verify_original_control(rows,refs,bad,anchor_m=[0,0,0],K_by_frame={f:K for f in ev.SCHEDULE},image_size=[640,480]);self.assertFalse(result['passed'])
 def test_rgb_decode_binding_uses_actual_pixels_in_RGB_order(self):
  from bench.workbench_geometry import array_sha256
  bgr=np.array([[[7,11,19],[23,29,31]]],np.uint8)
  self.assertEqual(ev.decode_pixel_sha256(b'fake',decode=lambda raw:bgr),array_sha256(np.ascontiguousarray(bgr[:,:,::-1])))
  self.assertNotEqual(ev.decode_pixel_sha256(b'fake',decode=lambda raw:bgr),array_sha256(bgr))
 def test_import_is_cpu_only_without_accepted_evaluator_or_torch(self):
  s="import sys;from tools import workbench_mask_replay_evaluate;assert 'torch' not in sys.modules and 'tools.workbench_ycb_dataset_evaluate' not in sys.modules"
  p=subprocess.run([sys.executable,'-B','-c',s],cwd=ev.ROOT,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)



class ProvenanceTests(unittest.TestCase):
 def terminal(self):
  h='a'*64
  return {'schema':ev.SCHEMA_TERMINAL,'run_id':ev.NEW_RUN,'execution_id':'exec8461','command':'recorded GPU command','exit_code':0,'worker_process_exited':True,'observed_by':'Codex parent','observed_at':'2026-10-21T10:17:00Z','attested':{'geometry':{'glb_path':'control/model.glb','glb_sha256':h,'physical_bytes_hashed_by_parent':True},'mask_inventory':{'manifest_path':'mask/manifest.json','manifest_file_sha256':h,'manifest_canonical_sha256':h,'frames':120,'inventory_sha256':h,'physical_files_hashed_by_parent':True},'inference_execution':{'launch_path':'launch.py','launch_sha256':h,'admission_dir':'admission/attempt-06','admission_file_hashes':{'contract.json':h},'report_dependencies_sha256':h,'surface_control_adapter_sha256':h,'observation_receipts':[{'path':'receipt.json','sha256':h}]}}}
 def test_terminal_requires_integer_zero_real_exit_and_nonempty_observation(self):
  record=self.terminal();ev.validate_terminal_record(record)
  for key,values in [('exit_code',[False,0.,'0',None,1]),('worker_process_exited',[False,1]),('run_id',['other']),('observed_by',['',' ']),('execution_id',[''])]:
   for value in values:
    bad=copy.deepcopy(record);bad[key]=value
    with self.subTest(key=key,value=value),self.assertRaises(ev.ReplayError):ev.validate_terminal_record(bad)
 def test_missing_geometry_attestation_or_duplicate_receipt_is_rejected(self):
  r=self.terminal();r['attested']['geometry']['physical_bytes_hashed_by_parent']=False
  with self.assertRaises(ev.ReplayError):ev.validate_terminal_record(r)
  r=self.terminal();i=r['attested']['inference_execution'];i['observation_receipts']*=2
  with self.assertRaises(ev.ReplayError):ev.validate_terminal_record(r)
 def mask_fixture(self):
  from bench.workbench_contract import canonical_sha256
  h='a'*64;producer={'kind':'test-observed-SAM'}
  m={'schema':ev.MASK_SCHEMA,'clip_id':ev.CLIP,'mask_set':ev.NEW_MASK_SET,'producer':producer,'evaluator_inputs_read':False,'masks':{f:{'path':f'm{i:05d}.png','sha256':h,'content_sha256':h} for i,f in enumerate(ev.SCHEDULE)}}
  rows={role:{f:{'mask_state':'observed','mask_frame_id':f,'mask_sha256':h} for f in ev.SCHEDULE} for role in ev.ROLES}
  stage={'status':'completed','identity':{'mask_set':ev.NEW_MASK_SET,'manifest_sha256':canonical_sha256(m),'producer':producer}}
  return rows,m,stage
 def test_consumed_observed_mask_cannot_be_replaced_by_another_frame_or_hash(self):
  rows,m,stage=self.mask_fixture();ev.validate_mask_consumption(rows,m,stage,mask_set=ev.NEW_MASK_SET)
  for key,value in [('mask_frame_id','s000999'),('mask_sha256','b'*64)]:
   broken=copy.deepcopy(rows);broken[ev.ROLES[0]][ev.SCHEDULE[0]][key]=value
   with self.subTest(key=key),self.assertRaises(ev.ReplayError):ev.validate_mask_consumption(broken,m,stage,mask_set=ev.NEW_MASK_SET)
  bad=copy.deepcopy(stage);bad['identity']['manifest_sha256']='c'*64
  with self.assertRaises(ev.ReplayError):ev.validate_mask_consumption(rows,m,bad,mask_set=ev.NEW_MASK_SET)
 def test_unavailable_mask_stays_unavailable_without_fabricated_identity(self):
  rows,m,stage=self.mask_fixture();r=rows[ev.ROLES[0]][ev.SCHEDULE[0]];r['mask_state']='unavailable'
  with self.assertRaises(ev.ReplayError):ev.validate_mask_consumption(rows,m,stage,mask_set=ev.NEW_MASK_SET)
  r.update(mask_frame_id=None,mask_sha256=None);out=ev.validate_mask_consumption(rows,m,stage,mask_set=ev.NEW_MASK_SET)
  self.assertEqual(out['mask_state_counts'][ev.ROLES[0]],{'observed':109,'unavailable':1})
 def test_inventory_digest_is_bound_to_every_physical_file_hash(self):
  _,m,_=self.mask_fixture();before=ev.mask_inventory_digest(m);self.assertEqual(before['frames'],110)
  changed=copy.deepcopy(m);changed['masks'][ev.SCHEDULE[-1]]['sha256']='b'*64
  self.assertNotEqual(ev.mask_inventory_digest(changed)['inventory_sha256'],before['inventory_sha256'])
  changed['masks'][ev.SCHEDULE[-1]]['path']='../outside.png'
  with self.assertRaises(ev.ReplayError):ev.mask_inventory_digest(changed)
 def closure_fixture(self):
  fixed_configs={k:h for k,(_p,h) in ev.CONFIG_FILES.items()};h='a'*64
  plan={'pins':{'frozen_plan_sha256':ev.FROZEN_PLAN_SHA256,'accepted_evaluator_sha256':ev.ACCEPTED_EVALUATOR_SHA256,'new_mask_manifest_sha256':ev.NEW_MASK_MANIFEST_SHA256,'configs':fixed_configs},'original_run':{'files':{'report.json':h},'mask_manifest':'old/manifest.json','mask_manifest_file_sha256':h,'sidecar_sha256':h,'clip':{'path':'clip.json','sha256':h}},'new_run':{'mask_manifest':'new/manifest.json','mask_manifest_file_sha256':ev.NEW_MASK_MANIFEST_SHA256},'admission':{'file_pins':ev.ADMISSION_FILE_PINS},'geometry':{'control_dir':'control','reviewed_map_sha256':h,'bundle_sha256':h},'rgb':{f:{'path':f'rgb/{f}.png','file_sha256':h} for f in ev.SCHEDULE},'terminal_observation':{'sha256':ev.TERMINAL_OBSERVATION[1]},'sources':{'helper.py':h},'runtime':{'numpy':'frozen'}}
  snap={'frozen_plan_sha256':ev.FROZEN_PLAN_SHA256,'accepted_evaluator_sha256':ev.ACCEPTED_EVALUATOR_SHA256,'original_run_files':plan['original_run']['files'],'new_run_error_receipts':[],'original_mask_manifest_sha256':h,'new_mask_manifest_sha256':ev.NEW_MASK_MANIFEST_SHA256,'configs':fixed_configs,'admission':ev.ADMISSION_FILE_PINS,'reviewed_map_sha256':h,'clip_sha256':h,'control_json':{'metric-bundle.json':h,'execution-control-sidecar.json':h},'rgb_files':{b['path']:h for b in plan['rgb'].values()},'terminal_observation_sha256':ev.TERMINAL_OBSERVATION[1],'sources':plan['sources'],'runtime':plan['runtime']}
  return plan,snap
 def test_changed_config_bundle_map_rgb_or_runtime_is_not_adopted_as_seal_truth(self):
  plan,snap=self.closure_fixture();ev.validate_expected_closure(plan,snap)
  for key in ['configs','control_json','reviewed_map_sha256','clip_sha256','rgb_files','sources','runtime','original_mask_manifest_sha256','admission','terminal_observation_sha256']:
   bad=copy.deepcopy(snap);bad[key]={} if isinstance(bad[key],dict) else 'b'*64
   with self.subTest(key=key),self.assertRaises(ev.ReplayError):ev.validate_expected_closure(plan,bad)
 def test_closure_never_opens_mesh_observed_mask_or_model_bytes(self):
  plan,_=self.closure_fixture();seen=[]
  def trap(path):
   p=Path(path);seen.append(p)
   if p.suffix.lower() in ['.glb','.ply','.obj','.pt'] or ('masks' in p.parts and p.suffix.lower()=='.png'):self.fail('forbidden artifact open '+str(p))
   return 'a'*64
  with patch.object(ev,'sha256_file',side_effect=trap),patch.object(ev,'_run_files',return_value={}),patch.object(ev,'_error_receipts',return_value=[]),patch.object(ev,'source_closure',return_value={}),patch.object(ev,'runtime_identity',return_value={}):ev.closure_snapshot(plan)
  self.assertTrue(seen)
 def test_receipt_reference_cannot_open_models_or_sealed_reference_directory(self):
  for path in ['.cache/video-workbench/models/fake.json','.cache/video-workbench/derived/dataset-reference/eval-v1/fake.json']:
   with patch.object(ev,'sha256_file',side_effect=AssertionError('forbidden read')):
    with self.assertRaises(ev.ReplayError):ev.hash_receipts([{'path':path,'sha256':'a'*64}])

class PreflightAndMutationTests(unittest.TestCase):
 def run_stubbed(self,fault):
  from contextlib import ExitStack
  refs,rows,sealed,frozen=fixture();plan={'tool_sha256':'b'*64,'anchor_m':[0,0,0],'K_by_frame':{f:K for f in ev.SCHEDULE},'image_size':[640,480]}
  seal={'terminal_record':{'path':'fake-terminal.json','sha256':'c'*64},'closure':{'same':True},'run_authentication':{}}
  hashes={'plan':'a'*64,'seal':'b'*64,'terminal':'c'*64}
  if fault.startswith('pre_'):hashes[fault[4:]]='d'*64
  after=dict(hashes)
  if fault.startswith('post_'):after[fault[5:]]='d'*64
  calls=[];published=[]
  def read(path,sha,what):
   calls.append(what)
   if what=='seal':return seal
   if what=='terminal record':return {'typed':'record'}
   if what=='sealed reference result':return sealed
   if what=='frozen plan':return frozen
   self.fail('unexpected read '+what)
  with ExitStack() as stack:
   for name,value in [('check_out',lambda p:Path(p)),('_load_plan',lambda *a:plan),('validate_seal',lambda *a:seal),('_read_pinned_json',read),('validate_terminal_record',lambda r:r),('closure_snapshot',lambda *a:seal['closure']),('validate_expected_closure',lambda *a:None),('authenticate_runs',lambda *a:{'summary':{},'original_rows':rows,'new_rows':rows}),('sha256_file',lambda *a:ev.SEALED_RESULT_SHA256),('check_budget',lambda *a,**k:None),('publish',lambda path,files:published.append(files))]:stack.enter_context(patch.object(ev,name,side_effect=value))
   budget=stack.enter_context(patch.object(ev,'preflight_eval_budget',side_effect=ev.ReplayError('budget blocked') if fault=='budget' else None,return_value=ev.MAX_COMBINED_BYTES))
   stack.enter_context(patch.object(ev,'_external_hashes',side_effect=[hashes,after]))
   with self.assertRaises(ev.ReplayError):ev.evaluate(Path('fake-plan.json'),'a'*64,Path('fake-seal.json'),'b'*64,Path('fake-out'))
  return calls,published,budget
 def test_budget_failure_is_before_reference_parser(self):
  calls,published,budget=self.run_stubbed('budget');budget.assert_called_once();self.assertNotIn('sealed reference result',calls);self.assertTrue(all(set(x)=={'failure.json'} for x in published))
 def test_pre_read_external_mutation_cannot_parse_reference(self):
  for item in ['plan','seal','terminal']:
   with self.subTest(item=item):
    calls,published,_=self.run_stubbed('pre_'+item);self.assertNotIn('sealed reference result',calls);self.assertTrue(all(set(x)=={'failure.json'} for x in published))
 def test_post_read_plan_seal_or_terminal_mutation_cannot_publish_comparison(self):
  for item in ['plan','seal','terminal']:
   with self.subTest(item=item):
    calls,published,_=self.run_stubbed('post_'+item);self.assertIn('sealed reference result',calls);self.assertTrue(published);self.assertTrue(all(set(x)=={'failure.json'} for x in published))
 def test_same_scored_inputs_generate_identical_json_and_csv(self):
  refs,rows,_,_=fixture();scores={role:ev.score_role(rows[role],refs,[0,0,0],{f:K for f in ev.SCHEDULE},[640,480]) for role in ev.ROLES}
  cells=ev.build_cells(scores,scores,rows,rows);self.assertEqual(len(cells),4);self.assertEqual(ev._dump(cells),ev._dump(copy.deepcopy(cells)));self.assertEqual(ev.render_csv(cells),ev.render_csv(copy.deepcopy(cells)))
 def test_unknown_visibility_does_not_change_reference_eligibility(self):
  refs,rows,_,_=fixture();refs[ev.SCHEDULE[0]]['visibility']='unknown';scores={role:ev.score_role(rows[role],refs,[0,0,0],{f:K for f in ev.SCHEDULE},[640,480]) for role in ev.ROLES}
  cells=ev.build_cells(scores,scores,rows,rows)
  for cell in cells.values():self.assertEqual(cell['all_reference_eligible']['denominator'],110);self.assertEqual(cell['fixed_visible']['denominator'],109)



class BackendAndArtworkBindingTests(unittest.TestCase):
 def identities(self):
  plan={'geometry':{'candidate_glb_sha256':'a'*64},'K_by_frame':{f:K.tolist() for f in ev.SCHEDULE}}
  original={}
  for role in ev.ROLES:
   ident={'profile':'workbench_integer_metric_v1','base_checkpoint_sha256':ev.BASE_SHA256,'installed_head_sha256':ev.HEAD_SHA256 if role=='primary_neural' else None,'installed_head_state_sha256':'b'*64,'install_digest':'c'*64,'admission_contract_sha256':ev.ADMISSION_CONTRACT_SHA256,'principal_point_origin':'integer_centers','initialization':'controlled opening pose','seed_frame_id':'s000370','seed_target_frame_id':'s000371','glb_sha256':'a'*64,'metric_units':'metres','effective_K':K.tolist(),'quality_claim':False}
   if role=='primary_neural':ident['export_sha256']=ev.HEAD_SHA256
   else:ident['head']='frozen_base_pose_head'
   original[role]={f:{'backend_identity':copy.deepcopy(ident)} for f in ev.SCHEDULE}
  return original,copy.deepcopy(original),plan
 def test_changed_profile_units_seed_preprocessing_or_model_installation_rejected(self):
  a,b,plan=self.identities();ev.validate_backend_identities(a,b,plan)
  fields={'profile':'different','metric_units':'relative','initialization':'other init','seed_frame_id':'s000369','seed_target_frame_id':'s000372','install_digest':'d'*64,'installed_head_state_sha256':'e'*64,'principal_point_origin':'corner_origin','effective_K':[[1001,0,320],[0,1000,240],[0,0,1]]}
  for key,value in fields.items():
   bad=copy.deepcopy(b)
   for row in bad[ev.ROLES[0]].values():row['backend_identity'][key]=value
   with self.subTest(key=key),self.assertRaises(ev.ReplayError):ev.validate_backend_identities(a,bad,plan)
 def test_missing_required_identity_cannot_equal_missing_and_pass(self):
  a,b,plan=self.identities()
  for table in [a,b]:
   for row in table[ev.ROLES[0]].values():del row['backend_identity']['profile']
  with self.assertRaises(ev.ReplayError):ev.validate_backend_identities(a,b,plan)
 def test_artwork_canonical_record_anchor_and_effective_K_are_bound(self):
  from bench.workbench_contract import canonical_sha256
  body={'schema':ev.ARTWORK_SCHEMA,'units':'metres','glb_sha256':'a'*64,'principal_point_origin':'integer_centers','effective_K':K.tolist(),'marker':{'face_id':2,'barycentric':[1.,0.,0.]},'mesh_faces':200}
  art={**body,'record_sha256':canonical_sha256(body)};plan={'geometry':{'candidate_glb_sha256':'a'*64},'artwork':{'record_sha256':art['record_sha256'],'face_id':2,'barycentric':[1.,0.,0.],'mesh_faces':200},'K_by_frame':{f:K.tolist() for f in ev.SCHEDULE}}
  ev.validate_artwork(art,plan,'fake')
  bad=copy.deepcopy(art);bad['marker']['barycentric']=[0.,1.,0.];bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
  with self.assertRaises(ev.ReplayError):ev.validate_artwork(bad,plan,'fake')
  bad=copy.deepcopy(art);bad['effective_K'][0][0]+=1;bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'});forged=copy.deepcopy(plan);forged['artwork']['record_sha256']=bad['record_sha256']
  with self.assertRaises(ev.ReplayError):ev.validate_artwork(bad,forged,'fake')

if __name__=='__main__':unittest.main()
