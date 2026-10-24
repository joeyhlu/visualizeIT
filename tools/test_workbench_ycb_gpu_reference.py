"""Independent arithmetic/state tests for the shared-backend UV experiment; no GL."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tools import workbench_ycb_gpu_reference as reference

class GPUReferenceTests(unittest.TestCase):
    def test_exact_official_rgba_required_for_every_hypothesis(self):
        rgba=np.zeros((2,3,4),np.uint8)
        sources={name:rgba.copy() for name in reference.HYPOTHESES}
        self.assertEqual(set(reference.require_texture_sources(sources,rgba)['hypotheses']),set(sources))
        for name in reference.HYPOTHESES:
            changed={key:value.copy() for key,value in sources.items()}
            changed[name][0,0,0]=1
            with self.assertRaises(reference.ProofError):reference.require_texture_sources(changed,rgba)
        for bad in [rgba.astype(np.float32),rgba[:,:,:3],None]:
            changed=dict(sources,candidate=bad)
            with self.assertRaises(reference.ProofError):reference.require_texture_sources(changed,rgba)

    def test_outside_derived_rejected_before_preflight_or_output(self):
        outside=reference.ROOT/'.cache/invalid-gpu-reference-output'
        self.assertFalse(outside.exists())
        with patch.object(reference,'preflight') as preflight:
            with self.assertRaises(reference.ProofError):
                reference.run(Path('unread-prepared'),outside,expected_prepared_sha256='0'*64)
            preflight.assert_not_called()
        self.assertFalse(outside.exists())
        with self.assertRaises(reference.ProofError):reference._check_budgets(reference.DERIVED_ROOT)

    def samples(self,n=128):
        return {'y':np.full(n,10,np.int64),'x':np.arange(1,n+1,dtype=np.int64),
                'face_id':np.zeros(n,np.int64),'barycentric':np.full((n,3),1/3,np.float64),
                'depth_mm':np.full(n,1000.,np.float64),'uv':np.full((n,2),.5,np.float64),
                'rgb':np.full((n,3),.5,np.float64)}

    def renders(self):
        return {name:(np.full((480,640,3),value,np.float32),np.full((480,640),1000.,np.float32))
                for name,value in [('reference',.5),('candidate',.5),('opposite',.8)]}

    def test_source_arrays_convert_units_once_and_keep_raw_uv_winding(self):
        points=np.array([[0.,0.,0.],[100.,0.,0.],[0.,50.,0.]])
        triangles=np.array([[0,2,1]])
        uv=np.array([[.1,.2],[.3,.4],[.5,.6]])
        got=reference.source_primitive_arrays(points,triangles,uv)
        np.testing.assert_array_equal(got['positions_m'],(points/1000).astype(np.float32)[[0,2,1]])
        np.testing.assert_array_equal(got['texcoord_0'],uv.astype(np.float32)[[0,2,1]])
        np.testing.assert_allclose(got['normals'],[[0,0,-1]]*3)

    def test_bad_source_indices_nonfinite_and_uv_shapes_rejected(self):
        points=np.array([[0.,0.,0.],[100.,0.,0.],[0.,50.,0.]])
        for p,t,u in [(points,np.array([[0,1,3]]),np.zeros((3,2))),
                      (points,np.array([[0.,1.,2.]]),np.zeros((3,2))),
                      (points,np.array([[0,1,2]]),np.zeros((3,1))),
                      (points*np.nan,np.array([[0,1,2]]),np.zeros((3,2)))]:
            with self.assertRaises(reference.ProofError):reference.source_primitive_arrays(p,t,u)

    def test_matching_reference_and_discriminated_opposite_pass(self):
        result=reference.compare_hypotheses(self.samples(),self.renders())
        self.assertEqual(result['status'],'pass');self.assertEqual(result['samples'],128)
        self.assertEqual(result['candidate_vs_reference']['depth_p99_mm'],0.)

    def test_missing_candidate_pixel_cannot_be_dropped(self):
        renders=self.renders();renders['candidate'][1][10,1]=0
        result=reference.compare_hypotheses(self.samples(),renders)
        self.assertEqual(result['status'],'fail')
        self.assertEqual(result['samples'],128)
        self.assertEqual(result['coverage']['candidate']['missing'],1)

    def test_nonfinite_and_insufficient_data_never_pass(self):
        renders=self.renders();renders['candidate'][0][10,1,0]=np.nan
        self.assertNotEqual(reference.compare_hypotheses(self.samples(),renders)['status'],'pass')
        self.assertEqual(reference.compare_hypotheses(self.samples(1),self.renders())['status'],'inconclusive')

    def test_per_channel_error_cannot_hide_in_pooled_rgb_median(self):
        renders=self.renders();renders['candidate'][0][:,:,2]+=.02
        result=reference.compare_hypotheses(self.samples(),renders)
        self.assertEqual(result['candidate_vs_reference']['rgb_median_pooled'],0.)
        self.assertEqual(result['status'],'fail')

    def test_p99_rgb_error_blocks_despite_zero_median(self):
        renders=self.renders();renders['candidate'][0][10,1:4,:]=1.
        result=reference.compare_hypotheses(self.samples(),renders)
        self.assertEqual(result['candidate_vs_reference']['rgb_median_pooled'],0.)
        self.assertEqual(result['status'],'fail')

    def test_ambiguous_opposite_cannot_pass(self):
        renders=self.renders();renders['opposite']=(renders['candidate'][0].copy(),renders['candidate'][1].copy())
        self.assertEqual(reference.compare_hypotheses(self.samples(),renders)['status'],'inconclusive')

    def test_one_failed_or_missing_frame_blocks_full_acceptance(self):
        self.assertIs(reference.relative_convention(['pass','pass']),True)
        self.assertEqual(reference.relative_convention(['pass','fail']),'fail')
        self.assertEqual(reference.relative_convention(['pass']),'inconclusive')
        self.assertEqual(reference.final_status(True,'inconclusive')[1],'inconclusive')

    def test_state_comparison_retains_sampler_differences(self):
        state={'mesh_nodes':1,'node_pose_identity':[True],'primitives':[{'material':{
            'base_color_texture':{'sampler':{'minFilter':9987,'wrapS':10497,'wrapT':10497}},
            'other_textures_present':{}}}]}
        states={name:copy.deepcopy(state) for name in reference.HYPOTHESES}
        self.assertTrue(reference.require_backend_equality(states)['equal_excluding_content'])
        states['candidate']['primitives'][0]['material']['base_color_texture']['sampler']['wrapT']=33071
        with self.assertRaises(reference.ProofError):reference.require_backend_equality(states)

if __name__=='__main__':unittest.main()
