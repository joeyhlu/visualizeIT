"""Portable lightweight tests: seals, causal input identities, metric denominators."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import tracking_offline_evaluate as ev


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,separators=(',',':')))


def native_run(roles=None):
    return dict(schema='workbench_run_v1',run_id='run-unit',clip_id='clip-unit',mode='regression',
        onboarding=dict(keyframe_ids=['f000009'],selection=dict(frame_id='f000009',positive=[[0,0]],negative=[[1,1]],reviewed=True,reviewed_by='unit_test_fixture')),
        scheduled_frame_ids=['f000010'],roles=roles if roles is not None else ['primary_neural'],
        primary=dict(artifact_id='unit-model',export_sha256='a'*64,export_bytes=1),surface_package_id=None,artwork_kind='checkerboard',config=dict(mask_set='unit-masks'))


def fixture(root, rows=None):
    if rows is None:rows=[]
    native=native_run()
    write(root/'run.json',dict(run=native));write(root/'report.json',dict(status='completed'))
    (root/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    seal=dict(schema='offline_recorded_prediction_seal_v1',evaluated=False,scoredGroundTruthOpenedBeforeSeal=False,
        run_id='run-unit',scoredFrameIDs=['f000010'],roles=['primary_neural'],rows=len(rows),files={})
    for name in ('run.json','report.json','results.jsonl'):
        p=root/name;seal['files'][name]=dict(bytes=p.stat().st_size,sha256=ev.digest(p))
    write(root/'seal.json',seal)
    return root/'seal.json',ev.digest(root/'seal.json')


def development_seal(role='primary_neural'):
    return dict(schema='offline_recorded_prediction_seal_v2',evaluated=False,
        evaluationProtocol='expanded_development_reused_reference',prior_reference_access=True,
        inference_process_scored_GT_access=False,scored_GT_feedback_to_inference=False,weights_math_thresholds_pinned=True,
        comparative_scoring_requires_all_role_seals=True,expectedComparativeRoles=['primary_neural','frozen_baseline'],
        roles=[role],scoredFrameIDs=['f000010'],nativeMaskManifestSHA256='a'*64)


def combined_pre_gt_fixture(root):
    """Manufactured tiny files, real native parsers/binding, NO models/GT."""
    from . import workbench_frame_boundary as fb
    from .workbench_neural import MetricBundle
    workspace=root/'workspace';run=workspace/'runs/run-unit';run.mkdir(parents=True)
    source=root/'inputs';(source/'rgb').mkdir(parents=True)
    frames=[]
    for i in (9,10):
        path=source/'rgb'/f'{i:06d}.png';Image.fromarray(np.full((2,2,3),i,dtype=np.uint8)).save(path)
        frames.append(dict(frame_id=f'f{i:06d}',source_frame_index=i,timestamp_s=None,media_time_s=None,image_path=path.name,image_sha256=ev.digest(path)))
    native=native_run(['primary_neural','frozen_baseline']);K=[2.,0.,.5,0.,2.,.5,0.,0.,1.]
    clip=dict(schema='workbench_clip_v1',clip_id='clip-unit',title='manufactured unit test, not model results',source_kind='image_sequence',clock_kind='frame_index',
        clock_source='unit source indices',width=2,height=2,media=dict(root_id='unit_inputs',path='rgb',sha256=None,bytes=None),frames=frames,
        intrinsics=dict(K=K,status='dataset_reference',source='manufactured unit fixture'),onboarding=native['onboarding'],scoring_frame_ids=['f000010'],
        evidence_role='manufactured_unit_test_fixture',evaluator_inputs_read=False,coordinates=dict(ev.wc.COORDINATE_CONVENTION))
    write(workspace/'clips/clip-unit/clip.json',clip);write(workspace/'config.json',dict(schema='workbench_config_v1',roots=dict(unit_inputs=str(source))))
    # Minimal embedded identity-node metre geometry; no texture/rendering.
    points=np.array([[0,0,0],[.01,0,0],[0,.01,0]],dtype='<f4');binary=points.tobytes()
    doc=dict(asset=dict(version='2.0'),nodes=[dict(mesh=0)],meshes=[dict(primitives=[dict(attributes=dict(POSITION=0))])],
        accessors=[dict(bufferView=0,componentType=5126,type='VEC3',count=3)],bufferViews=[dict(buffer=0,byteOffset=0,byteLength=len(binary))],buffers=[dict(byteLength=len(binary))])
    body=json.dumps(doc).encode();body+=b' '*((-len(body))%4);model=root/'model.glb'
    model.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(body)+len(binary))+struct.pack('<I4s',len(body),b'JSON')+body+struct.pack('<I4s',len(binary),b'BIN\0')+binary)
    seed=np.eye(4);seed[2,3]=1
    bundle=dict(schema='workbench_metric_bundle_v1',family='unit_family',glb_path=str(model),glb_sha256=ev.digest(model),obj_id=20,
        principal_point_origin='integer_centers',seed_pose_m=seed.tolist(),seed_label='onboarding:manufactured_unit_seed',seed_frame_id='f000009',metric_provenance='dataset_metric_cad')
    bp=workspace/'metric-bundles/clip-unit.json';write(bp,bundle)
    binding=fb.neural_binding(ev.wc.ClipSpec.from_dict(clip),ev.wc.RunSpec.from_dict(native),family='unit_family',bundle=MetricBundle.from_file(bp))
    mask_path=workspace/'masks/clip-unit/unit-masks/m00001.png';mask_path.parent.mkdir(parents=True);Image.fromarray(np.zeros((2,2),np.uint8)).save(mask_path)
    mask_spec=dict(path=mask_path.name,sha256=ev.digest(mask_path),content_sha256=ev.mask_sha256(np.zeros((2,2),bool)),empty=True,width=2,height=2)
    manifest=dict(schema='workbench_observed_masks_v1',clip_id='clip-unit',mask_set='unit-masks',cancelled=False,evaluator_inputs_read=False,
        producer=dict(kind='sam2.1_video_prompted',note='manufactured parser fixture, no actual SAM execution'),masks=dict(f000010=mask_spec))
    write(mask_path.parent/'manifest.json',manifest);rows=[]
    rgb_sha=ev.array_sha256(np.full((2,2,3),10,dtype=np.uint8))
    for role in native['roles']:
        identity=dict(role=role,metric_units='metres',principal_point_origin='integer_centers',effective_K=K,glb_sha256=bundle['glb_sha256'],
            seed_frame_id='f000009',seed_target_frame_id='f000010',execution_scope='finite_recorded_only',live_eligible=False)
        rows.append(ev.wc.FrameResult('run-unit','clip-unit','f000010',10,None,None,role,identity,'observed','lost','suppressed','observed_mask_empty',None,None,rgb_sha,'f000010',mask_spec['content_sha256'],{}).to_dict())
    write(run/'run.json',dict(run=native));(run/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write(run/'report.json',dict(status='completed',dependencies=dict(sources=dict(unit_fixture='0'*64)),
        stages={role:dict(execution=dict(binding_sha256=ev.wc.canonical_sha256(binding))) for role in native['roles']}))
    seal=development_seal();seal.update(run_id='run-unit',rows=2,roles=native['roles'],nativeMaskManifestSHA256=ev.wc.canonical_sha256(manifest),files={})
    for name in ('run.json','report.json','results.jsonl'):seal['files'][name]=dict(bytes=(run/name).stat().st_size,sha256=ev.digest(run/name))
    sp=run/'prediction-seal.json';write(sp,seal)
    return workspace,run,sp,ev.digest(sp)


class SealingTests(unittest.TestCase):
    def test_native_combined_run_parses_and_baseline_only_is_rejected(self):
        combined=native_run(['primary_neural','frozen_baseline'])
        self.assertEqual(ev.wc.RunSpec.from_dict(combined).roles,('primary_neural','frozen_baseline'))
        with self.assertRaisesRegex(ValueError,'comparator and cannot replace'):ev.wc.RunSpec.from_dict(native_run(['frozen_baseline']))

    def test_standalone_baseline_seal_rejected_by_native_contract_before_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,_=fixture(root);write(root/'run.json',dict(run=native_run(['frozen_baseline'])))
            seal=ev.load(path);seal['roles']=['frozen_baseline'];seal['files']['run.json']=dict(bytes=(root/'run.json').stat().st_size,sha256=ev.digest(root/'run.json'));write(path,seal)
            with self.assertRaisesRegex(ValueError,'comparator and cannot replace'):ev.verify_seal(root,path,ev.digest(path))

    def test_valid_seal_with_missing_result_retains_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,sha=fixture(root);seal,native,rows=ev.verify_seal(root,path,sha)
            self.assertEqual(rows,[]);self.assertEqual(native['scheduled_frame_ids'],['f000010'])

    def test_modified_prediction_fails_before_evaluator_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,sha=fixture(root);(root/'results.jsonl').write_text('tampered')
            with patch.object(ev,'verified_inputs') as inputs,patch.object(ev,'load',wraps=ev.load) as reads:
                with self.assertRaisesRegex(ValueError,'identity differs'):ev.evaluate(root,root,path,sha,root/'NONEXISTENT_GT',root)
                inputs.assert_not_called()
                self.assertFalse(any('evaluator' in str(call.args[0]) for call in reads.call_args_list))

    def test_external_seal_digest_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,sha=fixture(root)
            with self.assertRaisesRegex(ValueError,'identity differs'):ev.verify_seal(root,path,'0'*64)

    def test_preseal_gt_access_attestation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,_=fixture(root);seal=ev.load(path);seal['scoredGroundTruthOpenedBeforeSeal']=True;write(path,seal)
            with self.assertRaisesRegex(ValueError,'isolation'):ev.verify_seal(root,path,ev.digest(path))

    def test_traversal_seal_resource_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,_=fixture(root);seal=ev.load(path);seal['files']['../outside']=dict(bytes=1,sha256='a'*64);write(path,seal)
            with self.assertRaisesRegex(ValueError,'escapes'):ev.verify_seal(root,path,ev.digest(path))

    def test_expanded_development_requires_explicit_opt_in(self):
        seal=development_seal()
        with self.assertRaisesRegex(ValueError,'explicit opt-in'):ev.reference_protocol(seal)
        self.assertEqual(ev.reference_protocol(seal,True)['mode'],'expanded_development_reused_reference')

    def test_expanded_development_rejects_false_globally_heldout_claim(self):
        for value in (False,True):
            seal=development_seal();seal['scoredGroundTruthOpenedBeforeSeal']=value
            with self.assertRaisesRegex(ValueError,'globally unopened'):ev.reference_protocol(seal,True)

    def test_expanded_development_contradictions_are_rejected(self):
        for key in ('prior_reference_access','inference_process_scored_GT_access','scored_GT_feedback_to_inference','weights_math_thresholds_pinned','comparative_scoring_requires_all_role_seals'):
            seal=development_seal();seal[key]=not seal[key]
            with self.assertRaisesRegex(ValueError,'Contradictory'):ev.reference_protocol(seal,True)
        seal=development_seal();seal['prior_reference_access']=1
        with self.assertRaisesRegex(ValueError,'Contradictory'):ev.reference_protocol(seal,True)

    def test_historical_v1_is_not_rewritten_as_untouched_reference_history(self):
        seal=dict(schema='offline_recorded_prediction_seal_v1',evaluated=False,scoredGroundTruthOpenedBeforeSeal=False)
        result=ev.reference_protocol(seal)
        self.assertEqual(result['mode'],'historical_v1_run_local_isolation');self.assertIn('NOT proof',result['note'])
        seal['prior_reference_access']=True
        with self.assertRaisesRegex(ValueError,'Mixed historical'):ev.reference_protocol(seal)

    def test_cohort_helper_requires_complete_unique_roles_not_native_admission(self):
        # Cohort-set math alone does not authorize standalone baseline runs.
        # Native RunSpec authority is separately exercised end-to-end below.
        primary=development_seal();baseline=development_seal('frozen_baseline')
        with self.assertRaisesRegex(ValueError,'BOTH unique'):ev.verify_comparison_seals(primary,[])
        with self.assertRaisesRegex(ValueError,'BOTH unique'):ev.verify_comparison_seals(primary,[primary])
        ev.verify_comparison_seals(primary,[baseline])
        baseline['nativeMaskManifestSHA256']='b'*64
        with self.assertRaisesRegex(ValueError,'source schedule, observed masks'):ev.verify_comparison_seals(primary,[baseline])

    def test_one_combined_v2_seal_covers_both_roles_without_companion(self):
        combined=development_seal();combined['roles']=['primary_neural','frozen_baseline']
        ev.reference_protocol(combined,True);ev.verify_comparison_seals(combined,[])

    def test_combined_native_v2_end_to_end_reaches_gt_boundary_without_companion(self):
        class ReachedReferenceBoundary(Exception):pass
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);workspace,run,seal,sha=combined_pre_gt_fixture(root);pack=root/'FORBIDDEN_REFERENCE'
            original_load=ev.load;reads=[]
            def guard(path):
                reads.append(Path(path))
                if Path(path)==pack/'LOCK.json':raise ReachedReferenceBoundary()
                return original_load(path)
            with patch.object(ev,'load',side_effect=guard):
                with self.assertRaises(ReachedReferenceBoundary):
                    ev.evaluate(workspace,run,seal,sha,pack,root/'FORBIDDEN_MASKS',allow_reused_reference_development=True)
            self.assertEqual(reads[-1],pack/'LOCK.json')
            self.assertFalse(any('evaluator.json' in str(path) for path in reads))

    def test_missing_comparison_seal_blocks_before_any_inputs_or_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,_=fixture(root);seal=ev.load(path);del seal['scoredGroundTruthOpenedBeforeSeal'];seal.update(development_seal());write(path,seal)
            with patch.object(ev,'verified_inputs') as inputs,patch.object(ev,'load',wraps=ev.load) as reads:
                with self.assertRaisesRegex(ValueError,'BOTH unique'):ev.evaluate(root,root,path,ev.digest(path),root/'NONEXISTENT_GT',root,allow_reused_reference_development=True)
                inputs.assert_not_called();self.assertFalse(any('evaluator' in str(call.args[0]) for call in reads.call_args_list))

    def test_v2_still_checks_all_prediction_file_hashes_before_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,_=fixture(root);seal=ev.load(path);del seal['scoredGroundTruthOpenedBeforeSeal'];seal.update(development_seal());write(path,seal)
            (root/'results.jsonl').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'identity differs'):ev.verify_seal(root,path,ev.digest(path),True)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.points=np.array([[0,0,0],[.02,0,0],[0,.02,0]],dtype=np.float64)
        self.reference=np.eye(4);self.reference[2,3]=1
        self.K=np.array([[100,0,320],[0,100,240],[0,0,1]],dtype=np.float64)

    def test_known_translation_native_pixels_and_metres(self):
        pred=self.reference.copy();pred[0,3]=.02;result=ev.pose_errors(pred,self.reference,self.points,self.K)
        self.assertAlmostEqual(result['mean_model_reprojection_px'],2)
        self.assertAlmostEqual(result['ADD_m'],.02);self.assertAlmostEqual(result['translation_m'],.02)
        self.assertAlmostEqual(result['rotation_deg'],0)

    def test_prediction_reflection_or_scale_rejected(self):
        for value in (-1,1.00001):
            pred=self.reference.copy();pred[0,0]=value
            with self.assertRaisesRegex(ValueError,'proper orthonormal'):ev.pose_errors(pred,self.reference,self.points,self.K)

    def test_reference_serialization_noise_is_not_invalid_prediction(self):
        reference=self.reference.copy();reference[0,0]+=9e-7
        self.assertGreater(ev.rotation_residuals(reference)['max_abs_RtR_minus_I'],1e-6)
        with self.assertRaises(ValueError):ev.validate_pose(reference)  # prediction audit is unchanged/strict
        result=ev.pose_errors(self.reference,reference,self.points,self.K)
        self.assertLess(result['mean_model_reprojection_px'],.00001)
        self.assertGreater(result['ADD_m'],0)  # reference was NOT silently orthogonalized

    def test_reference_exceeding_existing_native_contract_is_rejected(self):
        reference=self.reference.copy();reference[0,0]+=1e-3
        with self.assertRaisesRegex(ValueError,'reference pose rotation'):ev.pose_errors(self.reference,reference,self.points,self.K)

    def test_negative_projection_depth_is_not_scored_as_correct(self):
        pred=self.reference.copy();pred[2,3]=-1
        with self.assertRaisesRegex(ValueError,'behind camera'):ev.pose_errors(pred,self.reference,self.points,self.K)

    def test_mask_overlap_and_empty_reference_false_foreground(self):
        a=np.array([[1,1],[0,0]]);b=np.array([[1,0],[1,0]])
        result=ev.mask_metrics(a,b);self.assertAlmostEqual(result['iou'],1/3);self.assertEqual(result['dice'],.5)
        self.assertEqual(result['reference_pixel_recall'],.5);self.assertEqual(result['predicted_pixel_precision'],.5)
        self.assertFalse(result['iou_at_least_0_5'])
        false_foreground=ev.mask_metrics(a,np.zeros_like(a));self.assertTrue(false_foreground['empty_reference_false_foreground'])
        self.assertIsNone(false_foreground['reference_pixel_recall']);self.assertEqual(false_foreground['iou'],0)
        empty=ev.mask_metrics(np.zeros_like(a),np.zeros_like(a));self.assertEqual(empty['iou'],1);self.assertFalse(empty['iou_at_least_0_5'])

    def test_missing_prediction_is_failure_not_invented_empty_perfect_prediction(self):
        result=ev.mask_metrics(None,np.zeros((2,2),bool))
        self.assertFalse(result['prediction_available']);self.assertEqual(result['iou'],0);self.assertEqual(result['dice'],0)
        self.assertIsNone(result['predicted_pixels']);self.assertFalse(result['empty_prediction']);self.assertFalse(result['iou_at_least_0_5'])

    def test_failed_missing_and_threshold_equal_outputs_stay_denominator(self):
        schedule=['f000010','f000011','f000012','f000013'];seg={fid:dict(reference_visibility=1,iou_at_least_0_5=True) for fid in schedule}
        errors=dict(mean_model_reprojection_px=4.9,ADD_m=.1,rotation_deg=2,translation_m=.03)
        good=dict(frame_id='f000011',pose_state='tracking',render_state='unavailable',valid_public_pose=True,pose_error=errors,reference_visibility=1,pose=np.eye(4).tolist(),metrics={})
        equal=copy.deepcopy(good);equal['frame_id']='f000012';equal['pose_error']['mean_model_reprojection_px']=5
        lost=dict(frame_id='f000010',pose_state='recovering',render_state='suppressed',valid_public_pose=False,pose_error=None,reference_visibility=1,pose=None,metrics={})
        summary,rows=ev.summarize_role([lost,good,equal],seg,schedule)
        self.assertEqual(summary['scheduled_frames'],4);self.assertEqual(summary['missing_rows'],1)
        self.assertEqual(summary['valid_public_poses'],2);self.assertEqual(summary['all_frame_reprojection_coverage'],.25)
        self.assertEqual(summary['conditional_public_pose_errors']['mean_model_reprojection_px']['count'],2)
        self.assertEqual(summary['pose_state_counts'],dict(recovering=1,tracking=2,missing_result=1))
        self.assertFalse(rows[-1]['qualified_reprojection']);self.assertEqual(len(rows),4)


class GeometryAndInstrumentationTests(unittest.TestCase):
    def test_embedded_original_vertex_sample_is_deterministic_and_bounded(self):
        points=np.arange(1800,dtype='<f4').reshape(600,3)/10000
        doc=dict(nodes=[dict(mesh=0)],meshes=[dict(primitives=[dict(attributes=dict(POSITION=0))])],
            accessors=[dict(bufferView=0,componentType=5126,type='VEC3',count=600)],bufferViews=[dict(buffer=0,byteOffset=0,byteLength=points.nbytes)])
        body=json.dumps(doc).encode();body+=b' '*((-len(body))%4);binary=points.tobytes()
        raw=struct.pack('<4sII',b'glTF',2,28+len(body)+len(binary))+struct.pack('<I4s',len(body),b'JSON')+body+struct.pack('<I4s',len(binary),b'BIN\0')+binary
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'mesh.glb';path.write_bytes(raw);sample,metadata=ev.model_vertices(path)
            self.assertEqual(sample.shape,(512,3));self.assertEqual(metadata['indices'][0],0);self.assertEqual(metadata['indices'][-1],599)
            np.testing.assert_array_equal(sample,points[np.linspace(0,599,512,dtype=int)].astype(np.float64))
            with self.assertRaises(ValueError):ev.model_vertices(path,513)

    def test_forward_counts_equal_full_refinement_calls_not_three_times_count(self):
        rows=[dict(backend_role=role,metrics=dict(iterations=5)) for role in ['primary_neural','frozen_baseline'] for _ in range(30)]
        counts={f'network_{index}':dict(backbone=150,decoder=150,pose_head=150) for index in range(2)}
        support=dict(pose={'observed-full-forward-counts.json':dict(record=counts)})
        result=ev.verify_forward_counts(rows,['primary_neural','frozen_baseline'],support)
        self.assertEqual(result['total_full_network_forwards'],300)
        counts['network_0']['decoder']=149
        with self.assertRaisesRegex(ValueError,'full-forward counts'):ev.verify_forward_counts(rows,['primary_neural','frozen_baseline'],support)

    def test_csv_and_json_replay_is_byte_identical_and_conflicts_refused(self):
        row=dict(backend_role='primary_neural',frame_id='f000010',source_frame_index=10,pose_state='recovering',render_state='suppressed',
            valid_public_pose=False,reference_visibility=1,qualified_reprojection=False,joint_mask_and_pose=False,pose_error=None,failure_reason='recovering')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=ev.write_outputs(root,dict(schema='unit'),[row],[dict(frame_id='f000010',iou=.9)],dict())
            b=ev.write_outputs(root,dict(schema='unit'),[row],[dict(frame_id='f000010',iou=.9)],dict())
            self.assertEqual(a,b)
            with self.assertRaisesRegex(ValueError,'Refusing'):ev.write_outputs(root,dict(schema='changed'),[row],[dict(frame_id='f000010',iou=.9)],dict())

    def test_forward_count_helper_only_single_role_is_not_native_runspec_authority(self):
        # Scalar hook aggregation is role-agnostic. This does NOT make a
        # frozen-baseline-only native RunSpec valid; its guard is tested above.
        rows=[dict(backend_role='frozen_baseline',metrics=dict()) for _ in range(300)]
        support=dict(pose={'observed-full-forward-counts.json':dict(record=dict(network_0=dict(backbone=0,decoder=0,pose_head=0)))})
        result=ev.verify_forward_counts(rows,['frozen_baseline'],support)
        self.assertEqual(result['total_full_network_forwards'],0);self.assertEqual(result['roles']['frozen_baseline']['scored_native_rows'],300)

    def test_state_only_loss_processing_is_not_model_attempt_speed(self):
        rows=[dict(frame_id='f000010',metrics=dict(iterations=5,end_to_end_latency_ms=6000,tracking_ms=5900)),
              dict(frame_id='f000011',metrics=dict(end_to_end_latency_ms=2,tracking_ms=1),failure_reason='recovery_unavailable: no validated prior')]
        result=ev.role_processing_latency(rows,dict(f000010=3000,f000011=4000))
        self.assertEqual(result['true_network_attempt_rows'],1);self.assertEqual(result['state_only_no_network_rows'],1)
        self.assertEqual(result['network_attempt_cached_mask_frame_ms']['mean'],6000)
        self.assertEqual(result['state_only_frame_ms']['mean'],2);self.assertEqual(result['total_native_refinement_forwards'],5)
        self.assertEqual(result['estimated_serial_SAM_plus_native_attempt_frame_ms']['mean'],9000)


class SeedAndMetadataTests(unittest.TestCase):
    seed_label='onboarding:single_authorized_frame'
    schedule=['f000010','f000011','f000012']

    def row(self,fid='f000010',reason='observed_mask_empty'):
        return dict(frame_id=fid,backend_role='primary_neural',pose=None,mask_state='observed',failure_reason=reason,
            mask_frame_id=fid,mask_sha256='a'*64,metrics={},evidence={})

    def check(self,rows,masks=None):
        return ev.verify_seed_usage(rows,['primary_neural'],self.schedule,masks if masks is not None else {'f000010':np.zeros((2,2),bool)},'f000010',self.seed_label)

    def test_empty_first_mask_can_prove_zero_seed_and_all_lost(self):
        rows=[self.row(),self.row('f000011','seed_not_applicable: seed only applies immediately after onboarding')]
        result=self.check(rows);self.assertEqual(result['primary_neural']['seed_uses'],0)
        seg={fid:dict(reference_visibility=1,iou_at_least_0_5=False,source_frame_index=int(fid[1:])) for fid in self.schedule}
        scored=[dict(row,pose_state='lost',render_state='suppressed',valid_public_pose=False,pose_error=None,reference_visibility=1) for row in rows]
        summary,all_rows=ev.summarize_role(scored,seg,self.schedule)
        self.assertEqual(summary['all_frame_reprojection_coverage'],0);self.assertEqual(summary['missing_rows'],1)
        self.assertEqual(summary['conditional_public_pose_errors']['ADD_m']['count'],0)
        self.assertEqual(all_rows[-1]['source_frame_index'],12)

    def test_explicit_null_initialization_receipt_in_no_init_path_is_safe(self):
        row=self.row();row['evidence']['initialization']=None
        self.assertEqual(self.check([row])['primary_neural']['seed_uses'],0)

    def test_unavailable_first_mask_requires_explicit_native_receipt(self):
        row=self.row(reason='observed_mask_unavailable');row.update(mask_state='unavailable',mask_frame_id=None,mask_sha256=None)
        with self.assertRaisesRegex(ValueError,'explicit initial'):self.check([row],{'f000010':None})
        row['evidence']['mask_unavailable_reason']='no observed mask for this frame'
        self.assertEqual(self.check([row],{'f000010':None})['primary_neural']['seed_uses'],0)

    def test_zero_seed_cannot_be_inferred_from_missing_first_row_or_unexplained_loss(self):
        with self.assertRaisesRegex(ValueError,'first-target'):self.check([self.row('f000011')])
        with self.assertRaisesRegex(ValueError,'explicit initial'):self.check([self.row(reason='unspecified failure')])
        with self.assertRaisesRegex(ValueError,'explicit initial'):self.check([self.row()],{'f000010':np.ones((2,2),bool)})

    def test_single_seed_must_be_first_target_with_actual_attempt(self):
        row=self.row(reason='recovering');row['evidence']['initialization']=self.seed_label;row['metrics']['iterations']=5
        mask=np.ones((2,2),bool);row['mask_sha256']=ev.mask_sha256(mask)
        self.assertEqual(self.check([row],{'f000010':mask})['primary_neural']['seed_uses'],1)
        late=copy.deepcopy(row);late['frame_id']='f000011'
        for rows in ([late],[row,late]):
            with self.assertRaisesRegex(ValueError,'Late/multiple'):self.check(rows)
        row['metrics']['iterations']=0
        with self.assertRaisesRegex(ValueError,'no actual'):self.check([row])

    def test_claimed_seed_attempt_cannot_use_empty_missing_or_stale_observed_mask(self):
        row=self.row(reason='recovering');row['evidence']['initialization']=self.seed_label;row['metrics']['iterations']=5
        for mask in (None,np.zeros((2,2),bool)):
            with self.assertRaisesRegex(ValueError,'contradicts empty/unavailable'):self.check([row],{'f000010':mask})
        with self.assertRaisesRegex(ValueError,'fresh same-frame'):self.check([row],{'f000010':np.ones((2,2),bool)})

    def test_zero_seed_path_cannot_hide_late_network_attempt_or_gt_reset(self):
        first=self.row();later=self.row('f000011');later['metrics']['iterations']=5
        with self.assertRaisesRegex(ValueError,'network attempt'):self.check([first,later])
        later['metrics']={};later['evidence']['initialization']='GT:future_scored_pose'
        with self.assertRaisesRegex(ValueError,'Unknown/GT'):self.check([first,later])

    def test_dynamic_object_scene_profile_schedule_roles_and_training_status(self):
        frames=[dict(frame_id=f'f{i:06d}',source_frame_index=i) for i in range(9,310)]
        schedule=[f'f{i:06d}' for i in range(10,310)]
        clip=dict(clip_id='offline-hb-scene000002-object000007-full',frames=frames,provenance=dict(profile='full'),onboarding=dict(keyframe_ids=['f000009']))
        bundle=dict(obj_id=7,seed_frame_id='f000009');indexed=dict(clip_id='offline-hb-scene000002-object000007',scene_id='000002')
        native=dict(scheduled_frame_ids=schedule,roles=['primary_neural','frozen_baseline']);manifest=dict(masks={f['frame_id']:{} for f in frames})
        result=ev.protocol_metadata(clip,bundle,indexed,native,manifest)
        self.assertEqual(result['scored_frame_count'],300);self.assertEqual(result['native_role_count'],2)
        self.assertEqual(result['native_roles'],['primary_neural','frozen_baseline']);self.assertEqual(result['profile'],'full');self.assertEqual(result['scored_source_frame_bounds'],[10,309])
        self.assertEqual(result['object_id'],7);self.assertIn('not_audited',result['training_object_overlap_status'])
        self.assertIn('object7, 300 scored frames, 2 native role(s)',result['scope']);self.assertNotIn('30-frame',result['scope'])
        bundle['obj_id']=20;self.assertEqual(ev.protocol_metadata(clip,bundle,indexed,native,manifest)['training_object_overlap_status'],'documented_training_object_overlap')

    def test_unavailable_mask_receipt_preserves_identity_and_rejects_stale_inputs(self):
        frame=dict(source_frame_index=10);mask=np.ones((2,2),bool);row=self.row(reason='observed_mask_unavailable')
        row.update(source_frame_index=10,timestamp_s=None,media_time_s=None,image_sha256='b'*64,mask_state='unavailable',mask_frame_id=None,mask_sha256=None)
        row['evidence']['mask_unavailable_reason']='native mask decoder unavailable'
        ev.verify_row_input_identity(row,frame,'b'*64,mask)
        row['image_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'prediction RGB'):ev.verify_row_input_identity(row,frame,'b'*64,mask)
        row['image_sha256']='b'*64;row['mask_state']='observed';row['mask_frame_id']='f000009';row['mask_sha256']=ev.mask_sha256(mask)
        with self.assertRaisesRegex(ValueError,'observed mask'):ev.verify_row_input_identity(row,frame,'b'*64,mask)


if __name__=='__main__':unittest.main()
