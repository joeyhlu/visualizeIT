import unittest
import numpy as np
from .quality_contract import project
from .quality_contour_pose import refine
from .vision import cv2


class ContourProposalTests(unittest.TestCase):
    def setup_scene(self):
        vertices=np.array([[-.2,-.07,0],[.2,-.07,0],[.2,.07,0],[-.2,.07,0]])
        k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
        truth=np.eye(4);truth[:3,3]=[0,0,.7]
        pixels,_=project(vertices,truth,k)
        mask=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(mask,np.rint(pixels).astype(np.int32),255)
        seed=truth.copy();seed[0,3]=.008;seed[1,3]=.005
        return vertices,k,mask,truth,seed

    def test_fixed_3d_edges_correct_translation_without_annotation_inputs(self):
        vertices,k,mask,truth,seed=self.setup_scene();proposed,stats=refine(vertices,k,mask,seed)
        self.assertIsNotNone(proposed);self.assertFalse(stats['accepted_tracking_pose'])
        before=np.linalg.norm(project(vertices,seed,k)[0]-project(vertices,truth,k)[0],axis=1).mean()
        after=np.linalg.norm(project(vertices,proposed,k)[0]-project(vertices,truth,k)[0],axis=1).mean()
        self.assertLess(after,before*.4)

    def test_occluder_cut_does_not_pull_the_whole_model_to_the_hand(self):
        vertices,k,mask,truth,seed=self.setup_scene();mask[255:,270:370]=0
        proposed,stats=refine(vertices,k,mask,seed);self.assertIsNotNone(proposed)
        after=np.linalg.norm(project(vertices,proposed,k)[0]-project(vertices,truth,k)[0],axis=1).mean()
        self.assertLess(after,2.)

    def test_missing_or_localized_boundary_cannot_be_proposed_as_tracking(self):
        vertices,k,mask,truth,seed=self.setup_scene()
        proposed,stats=refine(vertices,k,np.zeros_like(mask),seed)
        self.assertIsNone(proposed);self.assertEqual(stats['reason'],'no_current_mask_boundary')
        tiny=np.zeros_like(mask);tiny[235:240,300:310]=255
        self.assertIsNone(refine(vertices,k,tiny,seed)[0])


if __name__=='__main__':unittest.main()
