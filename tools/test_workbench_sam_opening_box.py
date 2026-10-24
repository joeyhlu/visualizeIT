"""Codex checks for opening-box prompt math, identity and changed-seed qualification."""
import copy,contextlib,hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tools import workbench_sam_opening_box as box
from tools import workbench_sam_forward_isolation as iso

class PromptTests(unittest.TestCase):
 def setUp(self):self.doc=json.loads((box.ROOT/box.PROMPT_FILE).read_bytes())
 def test_frozen_prompt_bytes_and_occluder_negative_inside_are_legal(self):
  self.assertEqual(hashlib.sha256((box.ROOT/box.PROMPT_FILE).read_bytes()).hexdigest(),box.PROMPT_SHA256)
  box.validate_prompt(self.doc)
  x0,y0,x1,y1=self.doc['box_xyxy_integer_centres'];x,y=self.doc['negative_points'][0]
  self.assertTrue(x0<=x<=x1 and y0<=y<=y1)
 def test_bool_float_nan_reversed_outside_and_alternative_box_rejected(self):
  for coords in [[True,48,583,278],[388.,48,583,278],[388,48,float('nan'),278],[583,48,388,278],[-1,48,583,278],[388,48,640,278],[389,48,583,278]]:
   with self.subTest(coords=coords):
    d=copy.deepcopy(self.doc);d['box_xyxy_integer_centres']=coords
    with self.assertRaises(iso.IsolationError):box.validate_prompt(d)
 def test_changed_points_frame_rgb_and_conversion_rejected(self):
  changes=[('positive_points',[[414,110],[462,213],[544,88]]),('negative_points',[[521,165],[430,290],[350,210]]),('frame_id','s000371'),('local_index',True),('rgb_file_sha256','0'*64),('sam_box_xyxy',[389.,49.,584.,279.])]
  for key,value in changes:
   with self.subTest(key=key):
    d=copy.deepcopy(self.doc);d[key]=value
    with self.assertRaises(iso.IsolationError):box.validate_prompt(d)
 def test_actual_adapter_call_adds_half_once_and_preserves_api_flags(self):
  calls=[]
  predictor=SimpleNamespace(add_new_points_or_box=lambda state,**kw:calls.append((state,kw)))
  adapter=box.BoxSamAdapter.__new__(box.BoxSamAdapter);adapter.predictor=predictor;adapter.torch=SimpleNamespace(inference_mode=contextlib.nullcontext);adapter.calls=[]
  points=np.array([[414.5,110.5],[462.5,213.5],[545.5,88.5],[520.5,165.5],[430.5,290.5],[350.5,210.5]],np.float32)
  labels=np.array([1,1,1,0,0,0],np.int32);state={}
  adapter.add_prompt(state,9,points,labels)
  self.assertEqual(len(calls),1);actual=calls[0][1]
  np.testing.assert_array_equal(actual['box'],[388.5,48.5,583.5,278.5]);np.testing.assert_array_equal(actual['points'],points)
  self.assertEqual(actual['labels'].dtype,np.int32);self.assertEqual(actual['box'].dtype,np.float32)
  self.assertTrue(actual['normalize_coords']);self.assertTrue(actual['clear_old_points']);self.assertEqual(actual['obj_id'],1)
  self.assertEqual(adapter.calls,[box.expected_call_record()])
 def test_adapter_rejects_double_shift_and_wrong_start_before_predictor_call(self):
  adapter=box.BoxSamAdapter.__new__(box.BoxSamAdapter);adapter.calls=[]
  adapter.predictor=SimpleNamespace(add_new_points_or_box=lambda *a,**k:self.fail('predictor called'))
  adapter.torch=SimpleNamespace(inference_mode=contextlib.nullcontext);args=box.box_call_arguments()
  for index,points in [(10,args['points']),(9,args['points']+.5)]:
   with self.assertRaises(iso.IsolationError):adapter.add_prompt({},index,points,args['labels'])
 def test_cache_identity_changes_with_new_prompt_or_box(self):
  baseline=box.cache_identity()
  for symbol,value in [('PROMPT_SHA256','0'*64),('BOX_XYXY',(387,48,583,278))]:
   with patch.object(box,symbol,value):self.assertNotEqual(box.cache_identity()['sha256'],baseline['sha256'])
 def test_accepted_engine_source_is_pinned(self):
  self.assertEqual(box.verify_accepted_source(),iso.sha256_file(Path(iso.__file__)))
  with patch.object(box,'ACCEPTED_ISOLATION_SHA256','0'*64):
   with self.assertRaises(iso.IsolationError):box.verify_accepted_source()
 def test_module_import_does_not_load_torch_or_sam(self):
  script="import sys;from tools import workbench_sam_opening_box;assert 'torch' not in sys.modules and 'sam2' not in sys.modules"
  p=subprocess.run([sys.executable,'-B','-c',script],cwd=box.ROOT,capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)

class QualificationTests(unittest.TestCase):
 def setUp(self):
  self.seed=np.zeros((8,8),bool);self.seed[:2,:5]=True
  self.masks={i:self.seed.copy() for i in range(120)}
 def qualify(self,**kwargs):
  args={'actual_calls':[box.expected_call_record(),box.expected_call_record()],'duplicate_seed_equal':True,'preview_local':iso.PREVIEW_LOCAL,'positive':[(0,0)],'negative':[(7,7)]};args.update(kwargs)
  return box.qualify_box(self.masks,**args)
 def test_changed_seed_can_pass_without_original_seed_equality_or_fixed_area(self):
  with patch.object(iso,'qualify',side_effect=AssertionError('must not invoke original qualifier')):r=self.qualify()
  self.assertTrue(r['qualified']);self.assertFalse(r['original_seed_equality']['applicable']);self.assertEqual(r['new_seed']['area'],10)
  self.assertEqual(r['gates']['scored_consecutive_iou']['pairs'],109)
 def test_actual_two_calls_and_duplicate_seed_equality_are_required(self):
  for calls in [[],[box.expected_call_record()], [dict(box.expected_call_record(),normalize_coords=False)]*2]:
   self.assertFalse(self.qualify(actual_calls=calls)['qualified'])
  self.assertFalse(self.qualify(duplicate_seed_equal=False)['qualified'])
 def test_threshold_equality_passes_but_one_pixel_below_half_fails(self):
  half=np.zeros((8,8),bool);half[0,:5]=True
  self.masks[11]=half
  r=self.qualify();self.assertTrue(r['qualified']);self.assertEqual(r['gates']['scored_consecutive_iou']['min_observed'],.5)
  self.masks[11][0,4]=False;r=self.qualify()
  self.assertFalse(r['qualified']);self.assertEqual(r['gates']['scored_area']['frames_below'],[11])
 def test_scored_pairs_exclude_prefix_to_first_score_transition(self):
  self.masks[8][:]=False
  self.assertTrue(self.qualify()['qualified'])
 def test_all_frames_and_preview_identity_required(self):
  self.assertFalse(self.qualify(preview_local=iso.PREVIEW_LOCAL[:-1])['qualified'])
  del self.masks[119]
  with self.assertRaises(iso.IsolationError):self.qualify()
 def test_seed_points_and_disconnected_component_are_not_bypassed(self):
  self.assertFalse(self.qualify(positive=[(7,7)])['qualified'])
  self.assertFalse(self.qualify(negative=[(0,0)])['qualified'])
  seed=self.seed.copy();seed[7,7]=True;self.masks={i:seed.copy() for i in range(120)}
  self.assertFalse(self.qualify(negative=[(6,6)])['gates']['seed_largest_component']['passed'])
 def test_empty_seed_and_zero_union_are_failures_not_perfect_continuity(self):
  self.masks={i:np.zeros_like(self.seed) for i in range(120)};r=self.qualify(positive=[])
  self.assertFalse(r['qualified']);self.assertFalse(r['gates']['seed_nonempty']['passed']);self.assertEqual(r['gates']['scored_consecutive_iou']['min_observed'],0)
 def test_connected_component_threshold_is_inclusive(self):
  seed=np.zeros((12,12),bool);seed[:5,:4]=True;seed[4,3]=False;seed[11,11]=True
  self.masks={i:seed.copy() for i in range(120)};r=self.qualify(negative=[(10,10)])
  self.assertEqual(r['gates']['seed_largest_component']['value'],.95);self.assertTrue(r['qualified'])


class PlanAdmissionTests(unittest.TestCase):
 def fixture(self):
  original=json.loads((box.ROOT/'.cache/video-workbench/derived/sam-forward-isolation/plans/original-prompt-forward-isolated-v1/plan.json').read_bytes())
  return {'schema':box.PLAN_SCHEMA,'clip':original['clip'],'thresholds':copy.deepcopy(box.THRESHOLDS),'flags':copy.deepcopy(box.FLAGS),'preview':original['preview'],'prompt':{'sha256':box.PROMPT_SHA256,'prompt_id':box.PROMPT_ID,'box_xyxy':list(box.BOX_XYXY),'call':box.expected_call_record()},'cache_identity':box.cache_identity(),'original_plan':{'sha256':box.ORIGINAL_PLAN_SHA256},'accepted_isolation_sha256':box.ACCEPTED_ISOLATION_SHA256}
 def test_prompt_cache_and_source_control_tampering_rejected(self):
  good=self.fixture();box.validate_plan_shape(good)
  for field in ['thresholds','flags','prompt','cache_identity','original_plan','accepted_isolation_sha256']:
   bad=copy.deepcopy(good);bad[field]={} if isinstance(bad[field],dict) else '0'*64
   with self.subTest(field=field),self.assertRaises(iso.IsolationError):box.validate_plan_shape(bad)
 def test_missing_frame_or_stale_preview_cannot_be_admitted(self):
  bad=self.fixture();bad['clip']['frames'].pop()
  with self.assertRaises(iso.IsolationError):box.validate_plan_shape(bad)
  bad=self.fixture();bad['preview']['local'][-1]=118
  with self.assertRaises(iso.IsolationError):box.validate_plan_shape(bad)
 def test_external_hash_failure_happens_before_runtime_rebuild(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'plan.json';path.write_bytes(iso.canonical_bytes(self.fixture()))
   with patch.object(box,'build_plan',side_effect=AssertionError('runtime rebuild attempted')) as rebuild:
    with self.assertRaisesRegex(iso.IsolationError,'externally expected'):box.run(path,'0'*64)
    rebuild.assert_not_called()
 def test_noncanonical_and_duplicate_key_plan_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'plan.json';path.write_text(json.dumps(self.fixture(),indent=2),encoding='utf-8')
   with self.assertRaisesRegex(iso.IsolationError,'canonical'):box._load_plan(path,hashlib.sha256(path.read_bytes()).hexdigest())
   path.write_bytes(b'{"schema":"x","schema":"y"}')
   with self.assertRaises(iso.IsolationError):box._load_plan(path,hashlib.sha256(path.read_bytes()).hexdigest())

if __name__=='__main__':unittest.main()
