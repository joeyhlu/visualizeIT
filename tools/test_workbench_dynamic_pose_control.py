"""Independent manufactured controls; no model or scored data is used."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from tools import workbench_dynamic_pose_control as control

class DynamicPoseControlTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[100.,0.,1.],[0.,120.,1.],[0.,0.,1.]])
        self.P=np.eye(4);self.P[2,3]=1000.
        self.depth=np.full((3,3),1000.)
    def test_identical_pose_has_zero_flow_with_nonidentity_crop(self):
        crop=np.eye(4);crop[:3,:3]=control.rot_z(27.)
        result=control.oracle_flow(self.depth,self.K,crop,self.P,self.P)
        np.testing.assert_allclose(result['flow'],0.,atol=1e-12)
        self.assertEqual(result['support_pixels'],9)
    def test_known_native_translation_yields_positive_source_to_query_flow(self):
        target=self.P.copy();target[0,3]=20.
        result=control.oracle_flow(self.depth,self.K,np.eye(4),self.P,target)
        np.testing.assert_allclose(result['flow'][...,0],2.,atol=1e-12)
        np.testing.assert_allclose(result['flow'][...,1],0.,atol=1e-12)
    def test_crop_rotation_changes_known_displacement_axis_and_sign(self):
        crop=np.eye(4);crop[:3,:3]=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        target=self.P.copy();target[0,3]=20.
        result=control.oracle_flow(self.depth,self.K,crop,self.P,target)
        np.testing.assert_allclose(result['flow'][...,0],0.,atol=1e-12)
        np.testing.assert_allclose(result['flow'][...,1],2.4,atol=1e-12)
    def test_zero_depth_is_not_oracle_support(self):
        depth=self.depth.copy();depth[0,0]=0.
        result=control.oracle_flow(depth,self.K,np.eye(4),self.P,self.P)
        self.assertEqual(result['support_pixels'],8);self.assertFalse(result['support'][0,0])
    def test_mm_conversion_changes_translation_only_and_preserves_input(self):
        pose=np.eye(4);pose[:3,3]=[.01,-.02,.6]
        original=pose.copy();converted=control.metres_to_mm(pose)
        np.testing.assert_array_equal(pose,original)
        np.testing.assert_allclose(converted[:3,3],[10.,-20.,600.])
        np.testing.assert_array_equal(converted[:3,:3],np.eye(3))
    def test_fixed_perturbation_is_camera_frame_not_object_frame(self):
        pose=np.eye(4);pose[:3,:3]=control.rot_z(90.);pose[:3,3]=[.2,.3,.4]
        result=control.perturbed_pose(pose)
        np.testing.assert_allclose(result[:3,3],[.205,.296,.410])
        np.testing.assert_allclose(result[:3,:3],control.rot_z(2.)@control.rot_y(-2.)@control.rot_x(1.)@pose[:3,:3])
    def test_bad_solver_few_inliers_and_improper_pose_cannot_pass(self):
        pose=np.eye(4);pose[2,3]=1.;points=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]])
        good=control.oracle_iteration_check(1,24,pose,pose,self.K,points)
        self.assertTrue(good['passed'])
        bad=pose.copy();bad[0,0]=-1.
        for state,inliers,pred in [(0,24,pose),(1,23,pose),(1,24,bad)]:
            self.assertFalse(control.oracle_iteration_check(state,inliers,pred,pose,self.K,points)['passed'])
    def test_known_pose_error_and_all_checkpoint_accounting(self):
        pose=np.eye(4);pose[2,3]=1.;changed=pose.copy();changed[0,3]=.01
        dt,dr=control.pose_errors(changed,pose)
        self.assertAlmostEqual(dt,.01);self.assertAlmostEqual(dr,0.)
        result=control.checkpoint_errors(self.K,changed,pose,np.array([[0.,0.,0.],[0.,0.,1.]]))
        self.assertEqual(result['points'],2);self.assertEqual(result['of'],2)
        self.assertAlmostEqual(result['p95_px'],.975)
    def test_manufactured_preflight_rejects_all_four_faults(self):
        result=control.manufactured_preflight()
        self.assertTrue(result['passed'])
        self.assertTrue(result['variants']['correct']['passed_oracle_check'])
        self.assertEqual(sum(x['passed_oracle_check'] for x in result['variants'].values()),1)

    def test_render_and_neural_forward_budgets_are_enforced(self):
        counts=control.Counters()
        for _ in range(control.MAX_RENDERS):counts.take_render('template_renders')
        with self.assertRaises(control.ControlError):counts.take_render('query_renders')
        self.assertEqual(counts.renders,63)
        calls=[];network=control.CountingNetwork(lambda *args:calls.append(args),counts)
        for _ in range(control.MAX_PRIMARY_FORWARDS):network('synthetic')
        with self.assertRaises(control.ControlError):network('extra')
        self.assertEqual(len(calls),30)

    def test_crop_observer_restores_production_function_even_on_error(self):
        from types import SimpleNamespace
        sentinel=({'templates':SimpleNamespace(depths=np.ones((3,3)))},['camera'],[np.eye(4)])
        original=lambda **kwargs:sentinel
        module=SimpleNamespace(compute_gotrack_inputs_from_init_poses=original)
        observer=control.CropObserver(control.Counters(),SimpleNamespace(get_intrinsic_matrix=lambda _:np.eye(3)))
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with observer.attached(module):
                result=module.compute_gotrack_inputs_from_init_poses(init_poses_cam_from_model=np.eye(4))
                self.assertIs(result,sentinel)
                np.testing.assert_array_equal(observer.latest['init_pose_mm'],np.eye(4))
                raise RuntimeError('injected')
        self.assertIs(module.compute_gotrack_inputs_from_init_poses,original)
        self.assertIsNone(observer.latest)

    def test_wrong_external_plan_hash_stops_before_model_setup(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=control.ROOT/'.cache') as tmp:
            path=Path(tmp)/'plan.json';path.write_text('{}')
            with patch.object(control,'check_out',return_value=Path('unused-output')),\
                 patch.object(control,'_check_pins') as setup:
                with self.assertRaises(control.ControlError):control.run(path,'0'*64,Path('output'),'cuda')
                setup.assert_not_called()

    def test_slightly_nonorthogonal_seed_yields_proper_synthetic_target_only(self):
        raw=np.eye(4);raw[:3,:3]*=.999995;raw[:3,3]=[.1,-.2,.6]
        before=raw.copy();result=control.synthetic_target_pose(raw);target=result['pose_m']
        np.testing.assert_array_equal(raw,before)
        np.testing.assert_array_equal(result['raw_seed_pose_m'],before)
        self.assertTrue(control.proper_pose(target))
        np.testing.assert_array_equal(target[:3,3],raw[:3,3])
        self.assertGreater(result['projection_max_abs_delta'],0.)
        self.assertLess(control.pose_errors(target,target)[1],.001)

    def test_reflection_and_excessive_seed_deformation_rejected(self):
        for factor in [-1.,.99]:
            raw=np.eye(4);raw[0,0]=factor
            with self.assertRaises(control.ControlError):control.synthetic_target_pose(raw)

    def test_diagonal_line_rejected_but_planar_support_accepted_at_any_scale(self):
        x=np.arange(32,dtype=float)
        line=np.stack([x,2*x,3*x],axis=1)
        result=control.support_diagnostics(line)
        self.assertFalse(result['accepted']);self.assertTrue(result['collinear'])
        xx,yy=np.meshgrid(np.arange(6),np.arange(6));plane=np.stack([xx.ravel(),yy.ravel(),np.zeros(36)],axis=1)
        for scale in [1e-5,1.,1e5]:
            result=control.support_diagnostics(plane*scale)
            self.assertTrue(result['accepted']);self.assertTrue(result['planar'])
            self.assertFalse(result['collinear'])

    def test_final_crop_inverse_and_mm_conversion_faults_are_detectable(self):
        native=np.eye(4);native[:3,3]=[.02,-.03,.6]
        crop=np.eye(4);crop[:3,:3]=control.rot_y(5.)
        crop_pose=crop@control.metres_to_mm(native)
        np.testing.assert_allclose(control.crop_to_native(crop_pose,crop),native,atol=1e-14)
        for mode in ['crop_not_inverted','omit_mm_to_m']:
            wrong=control.crop_to_native(crop_pose,crop,mode)
            self.assertGreater(control.pose_errors(wrong,native)[0],1e-5)

    def test_actual_nested_head_provenance_required_without_silent_fallback(self):
        provenance={'head':{'requested_sha256':control.HEAD_SHA256,'installed_sha256':control.HEAD_SHA256},
                    'head_installation':{'requested':True,'installed':True},
                    'artifact_manifest_sha256':control.RECORD_SHA256,'installed_head_state_sha256':'a'*64}
        self.assertEqual(control.validate_head_provenance(provenance)['installed_sha256'],control.HEAD_SHA256)
        import copy
        for field in ['head','head_installation','artifact_manifest_sha256']:
            bad=copy.deepcopy(provenance);bad.pop(field)
            bad['installed_head_sha256']=control.HEAD_SHA256
            with self.assertRaises(control.ControlError):control.validate_head_provenance(bad)

    def test_rotation_angle_stays_accurate_at_small_and_half_turn_angles(self):
        for angle in [.0001,2.,180.]:
            pose=np.eye(4);pose[:3,:3]=control.rot_z(angle)
            self.assertAlmostEqual(control.pose_errors(pose,np.eye(4))[1],angle,places=8)

    def test_rehashed_plan_with_changed_target_fails_authentic_reconstruction(self):
        import tempfile,json,hashlib
        with tempfile.TemporaryDirectory(dir=control.ROOT/'.cache') as tmp:
            path=Path(tmp)/'plan.json'
            frozen={'schema':control.SCHEMA_PLAN,'objects':{'2':{'synthetic_target':'authentic'}}}
            changed={'schema':control.SCHEMA_PLAN,'objects':{'2':{'synthetic_target':'substituted'}}}
            data=json.dumps(changed).encode();path.write_bytes(data)
            with patch.object(control,'check_out',return_value=Path('unused-output')),\
                 patch.object(control,'build_plan',return_value=frozen),\
                 patch.object(control,'_budget') as gpu_setup:
                with self.assertRaises(control.ControlError):control.run(path,hashlib.sha256(data).hexdigest(),Path('output'),'cuda')
                gpu_setup.assert_not_called()

    def test_missing_or_corrupt_base_checkpoint_cannot_use_expected_identity(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=control.ROOT/'.cache') as tmp:
            path=Path(tmp)/'base.pt'
            with self.assertRaises(control.ControlError):control.verify_base_checkpoint(path)
            path.write_bytes(b'incorrect weights')
            with self.assertRaises(control.ControlError):control.verify_base_checkpoint(path)

if __name__=='__main__':unittest.main()
