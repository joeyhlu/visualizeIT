"""Independent manufactured tests for post-run projection metrics; no reference data read."""
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from tools import workbench_ycb_dataset_evaluate as evaluation

class DatasetProjectionTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[1000.,0.,320.],[0.,1000.,240.],[0.,0.,1.]])
        self.T=np.eye(4);self.T[2,3]=1.
        self.X=np.array([0.,0.,0.])

    def row(self,T=None,**changes):
        result={'pose_state':'tracking','pose_units':'metric','pose':(self.T if T is None else T).tolist()}
        result.update(changes);return result

    def test_exact_pose_zero_error(self):
        result=evaluation.anchor_frame(self.X,self.K,self.T,self.row(),640,480)
        self.assertEqual(result['error_native_px'],0.);self.assertEqual(result['error_px720'],0.)

    def test_known_translation_and_720_scale(self):
        moved=self.T.copy();moved[0,3]=.01
        result=evaluation.anchor_frame(self.X,self.K,self.T,self.row(moved),640,480)
        self.assertAlmostEqual(result['error_native_px'],10.)
        self.assertAlmostEqual(result['error_px720'],15.)

    def test_mm_translation_converted_once_and_bad_rotation_not_repaired(self):
        pose=evaluation.bop_pose_m(np.eye(3).ravel(),[10.,0.,1000.])
        np.testing.assert_array_equal(pose[:3,3],[.01,0.,1.])
        self.assertIsNone(evaluation.bop_pose_m((np.eye(3)*2).ravel(),[0.,0.,1000.]))

    def test_invalid_prediction_remains_reference_eligible(self):
        for row in [None,self.row(pose_units='object_unit'),self.row(pose_state='recovering')]:
            result=evaluation.anchor_frame(self.X,self.K,self.T,row,640,480)
            self.assertEqual(result['reference_status'],'eligible');self.assertIsNone(result['error_px720'])

    def test_behind_camera_and_out_of_image_predictions_are_failures(self):
        for changed,status in [(-1.,'prediction_behind_camera'),(100.,'prediction_out_of_image')]:
            moved=self.T.copy()
            if changed<0:moved[2,3]=changed
            else:moved[0,3]=changed
            result=evaluation.anchor_frame(self.X,self.K,self.T,self.row(moved),640,480)
            self.assertEqual(result['prediction_status'],status);self.assertIsNone(result['error_px720'])

    def test_failure_aware_coverage_uses_full_eligible_denominator(self):
        result=evaluation.error_summary([0.,3.,None],4)
        self.assertEqual(result['reference_eligible'],4);self.assertEqual(result['errors_measured'],2)
        self.assertEqual(result['coverage_le_5px720'],.5)

    def test_reference_visibility_does_not_depend_on_prediction(self):
        reference=self.T.copy();reference[0,3]=100.
        for row in [None,self.row()]:
            result=evaluation.anchor_frame(self.X,self.K,reference,row,640,480)
            self.assertEqual(result['reference_status'],'reference_out_of_image')

    def test_samples_are_deterministic_surface_identities(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces=np.array([[0,1,2]])
        first=evaluation.cad_samples(vertices,faces,2);second=evaluation.cad_samples(vertices,faces,2)
        self.assertEqual(first['hashes'],second['hashes'])
        np.testing.assert_allclose(first['barycentric'].sum(axis=1),1.)
        np.testing.assert_allclose(first['positions_m'][:,2],0.)
        self.assertTrue((first['positions_m']>=0).all())
        self.assertNotEqual(first['hashes'],evaluation.cad_samples(vertices,faces,4)['hashes'])

    def test_anchor_barycentric_point_and_invalid_coordinates(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        np.testing.assert_allclose(evaluation.anchor_point(vertices,np.array([[0,1,2]]),0,[.2,.3,.5]),[.3,.5,0.])
        with self.assertRaises(evaluation.EvalError):evaluation.anchor_point(vertices,np.array([[0,1,2]]),0,[-.1,.4,.7])

    def test_reference_ray_detects_foreground_surface(self):
        vertices=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]])
        faces=np.array([[0,1,2],[3,4,5]])
        self.assertTrue(evaluation.first_hit_occludes(vertices,faces,np.eye(4),np.array([.1,.1,2.]),own_face=1))
        self.assertFalse(evaluation.first_hit_occludes(vertices,faces,np.eye(4),np.array([.1,.1,1.]),own_face=0))

    def test_depth_tolerance_and_mask_leakage_arithmetic(self):
        self.assertTrue(evaluation.depth_consistent(900.,908.))
        self.assertFalse(evaluation.depth_consistent(900.,910.))
        result=evaluation.mask_overlap(np.array([[1,1],[0,0]]),np.array([[0,1],[0,0]]))
        self.assertEqual(result['iou'],.5);self.assertEqual(result['sam_outside_reference_fraction'],.5)
        self.assertIsNone(evaluation.mask_overlap(None,None)['iou'])

    def test_identity_join_handles_reordering_and_rejects_missing_or_duplicate(self):
        scheduled=['f1','f2'];rows=[{'backend_role':role,'frame_id':fid} for role in evaluation.ROLES for fid in scheduled]
        joined=evaluation.join_role_rows(list(reversed(rows)),scheduled)
        self.assertEqual(joined['primary_neural']['f1']['frame_id'],'f1')
        for changed in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaises(evaluation.EvalError):evaluation.join_role_rows(changed,scheduled)

    def test_missing_terminal_seal_blocks_before_any_reference_read(self):
        with patch.object(evaluation,'check_out',return_value=Path('unused-output')),\
             patch.object(evaluation,'load_plan',return_value=({},Path('unused-plan'))),\
             patch.object(evaluation,'validate_seal',side_effect=evaluation.EvalError('missing terminal seal')),\
             patch.object(evaluation,'_evaluate_after_seal') as labels:
            with self.assertRaises(evaluation.EvalError):evaluation.evaluate(Path('plan'),'0'*64,Path('seal'),Path('output'))
            labels.assert_not_called()

    def test_changed_inputs_block_before_any_reference_read(self):
        with patch.object(evaluation,'check_out',return_value=Path('unused-output')),\
             patch.object(evaluation,'load_plan',return_value=({},Path('unused-plan'))),\
             patch.object(evaluation,'validate_seal',return_value=({},'1'*64)),\
             patch.object(evaluation,'input_closure',return_value={'artifact':'changed'}),\
             patch.object(evaluation,'expected_closure',return_value={'artifact':'original'}),\
             patch.object(evaluation,'_evaluate_after_seal') as labels:
            with self.assertRaises(evaluation.EvalError):evaluation.evaluate(Path('plan'),'0'*64,Path('seal'),Path('output'))
            labels.assert_not_called()

    def test_reference_visibility_missing_labels_remain_unknown(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces=np.array([[0,1,2]])
        with patch.object(evaluation,'first_hit_occludes',return_value=False):
            result=evaluation.anchor_visibility(self.X,vertices,faces,0,self.K,self.T,None,'missing',None,'missing',1.,640,480)
            self.assertEqual(result['status'],'unknown')
            result=evaluation.anchor_visibility(self.X,vertices,faces,0,self.K,self.T,np.ones((480,640)),'ok',None,'missing',1.,640,480)
            self.assertEqual(result['status'],'unknown')

    def test_cad_one_perfect_survivor_cannot_hide_255_failed_points(self):
        points=np.zeros((256,3));points[-1,2]=1.
        moved=np.eye(4);moved[2,3]=-.25
        result=evaluation.samples_frame(points,self.K,self.T,self.row(moved),480)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['reference_eligible_points'],256)
        self.assertEqual(result['paired_points'],1)
        self.assertAlmostEqual(result['paired_fraction'],1/256)
        self.assertEqual(result['conditional_median_px720'],0.)
        self.assertEqual(result['prediction_behind_camera_points'],255)
        self.assertAlmostEqual(result['coverage_le_5px720'],1/256)

    def test_cad_missing_pose_retains_all_reference_eligible_points(self):
        points=np.zeros((256,3))
        result=evaluation.samples_frame(points,self.K,self.T,None,480)
        self.assertEqual(result['reference_eligible_points'],256)
        self.assertEqual(result['paired_points'],0)
        self.assertEqual(result['coverage_le_5px720'],0.)
        self.assertIsNone(result['conditional_median_px720'])
        self.assertEqual(result['prediction_invalid_points'],256)

    def test_authenticated_malformed_label_preserves_unavailable_status(self):
        import tempfile,hashlib
        with tempfile.TemporaryDirectory(dir=evaluation.ROOT/'.cache') as tmp:
            path=Path(tmp)/'malformed.png';data=b'not a PNG despite its authenticated hash';path.write_bytes(data)
            reads={}
            array,status=evaluation._decode_label(path,hashlib.sha256(data).hexdigest(),'mask',reads)
            self.assertIsNone(array);self.assertNotEqual(status,'ok')
            self.assertTrue(reads)

    def test_rgb_binding_authenticates_file_and_exact_rgb_pixel_array(self):
        import tempfile,hashlib
        from PIL import Image
        from bench.workbench_geometry import array_sha256
        with tempfile.TemporaryDirectory(dir=evaluation.ROOT/'.cache') as tmp:
            folder=Path(tmp);path=folder/'rgb.png'
            rgb=np.zeros((480,640,3),dtype=np.uint8);rgb[:]=[17,39,201]
            Image.fromarray(rgb).save(path)
            ref={'available':True,'path':'rgb.png','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.object(evaluation,'DATASET_ROOT',folder):
                binding=evaluation.rgb_binding(ref)
                self.assertEqual(binding['pixel_sha256'],array_sha256(rgb))
                with self.assertRaises(evaluation.EvalError):evaluation.rgb_binding({**ref,'sha256':'0'*64})

    def test_helper_closure_contains_actual_mesh_contract_and_array_hash_sources(self):
        closure=evaluation.helper_sources()
        for source in ['bench/workbench_neural.py','bench/workbench_geometry.py','bench/workbench_contract.py']:
            self.assertEqual(closure[source],evaluation.sha256_file(evaluation.ROOT/source))

    def test_runtime_drift_blocks_before_reference_read(self):
        with patch.object(evaluation,'check_out',return_value=Path('unused-output')),\
             patch.object(evaluation,'load_plan',return_value=({'runtime':{'numpy':'frozen'}},Path('unused-plan'))),\
             patch.object(evaluation,'validate_seal',return_value=({},'1'*64)),\
             patch.object(evaluation,'input_closure',return_value={}),\
             patch.object(evaluation,'expected_closure',return_value={}),\
             patch.object(evaluation,'runtime_identity',return_value={'numpy':'changed'}),\
             patch.object(evaluation,'_evaluate_after_seal') as labels:
            with self.assertRaises(evaluation.EvalError):evaluation.evaluate(Path('plan'),'0'*64,Path('seal'),Path('output'))
            labels.assert_not_called()

if __name__=='__main__':unittest.main()
