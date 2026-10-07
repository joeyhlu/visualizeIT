import unittest
from unittest.mock import patch
import numpy as np
from .vision import cv2
from .model import Mesh
from .surface_tracker import SurfaceRays,SurfaceTracker,projected


def fixture():
    positions=np.array([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]])
    mesh=Mesh('plane',positions,np.tile([0,0,-1.],(4,1)),np.zeros((4,2)),np.array([[0,2,1],[0,3,2]]))
    x,y=np.meshgrid(np.linspace(-.08,.08,5),np.linspace(-.08,.08,5))
    points=np.column_stack((x.ravel(),y.ravel(),np.zeros(25)))
    k=np.array([[600.,0,320],[0,600,360],[0,0,1]])
    pose=(cv2.Rodrigues(np.array([.1,.2,0]))[0],np.array([0.,0,.8]))
    gray=np.zeros((720,640),np.uint8);pixels=projected(points,*pose,k)
    tracker=SurfaceTracker(points,np.random.default_rng(7).integers(0,256,(25,32),dtype=np.uint8),
        positions,gray,pixels,points,mesh=mesh,initial_pose=pose)
    return mesh,points,k,pose,tracker


class SurfaceTrackingTests(unittest.TestCase):
    def test_actual_surface_rays_preserve_rotated_camera_and_reject_empty_space(self):
        mesh,points,k,pose,_=fixture();rays=SurfaceRays(mesh)
        pixels=projected(points,*pose,k)
        hits=rays.intersect(pixels,k,*pose)
        np.testing.assert_allclose(hits,points,atol=1e-8)
        self.assertTrue(np.isnan(rays.intersect(np.array([[0.,0.]]),k,*pose)).all())

    def test_partial_occlusion_outliers_do_not_flip_a_planar_pose(self):
        _,points,k,pose,tracker=fixture()
        current=(cv2.Rodrigues(np.array([.11,.205,.005]))[0],pose[1]+[.003,-.002,.001])
        pixels=projected(points,*current,k)
        pixels[:8]+=np.random.default_rng(9).normal(0,40,(8,2))
        result,retained=tracker.estimate(pixels,points,k,1)
        self.assertEqual(result.state,'tracking')
        self.assertIsNotNone(retained)
        angle=np.arccos(np.clip((np.trace(result.rotation@current[0].T)-1)/2,-1,1))
        self.assertLess(angle,.02);self.assertLess(np.linalg.norm(result.translation-current[1]),.003)

    def test_weak_frame_hides_pose_but_preserves_flow_and_reacquires(self):
        _,points,k,pose,tracker=fixture()
        small=points[:3].copy();small[:,0]*=.15
        small=np.concatenate((small,small+[0,.01,0]))
        tracker.previous_points=small;tracker.previous_pixels=projected(small,*pose,k)
        def follow_identity(previous,current,pixels):return np.asarray(pixels).copy(),np.ones(len(pixels),bool)
        with patch('bench.surface_tracker.follow',side_effect=follow_identity),patch('bench.surface_tracker.geometry_corners',return_value=(np.empty((0,2)),np.empty((0,3)))):
            tracker.match=lambda gray:(np.empty((0,2)),np.empty((0,3)))
            weak=tracker.update(tracker.previous_gray,k,1)
            self.assertEqual(weak.state,'limited');self.assertIsNone(weak.rotation)
            self.assertGreaterEqual(len(tracker.previous_pixels),6)
            tracker.match=lambda gray:(projected(points,*pose,k),points)
            recovered=tracker.update(tracker.previous_gray,k,2)
            self.assertEqual(recovered.state,'tracking')

    def test_known_initial_pose_is_never_used_as_a_visible_fallback(self):
        _,_,k,_,tracker=fixture()
        tracker.previous_pixels=np.empty((0,2));tracker.previous_points=np.empty((0,3))
        tracker.match=lambda gray:(np.empty((0,2)),np.empty((0,3)))
        for i in range(15):
            result=tracker.update(tracker.previous_gray,k,i)
            self.assertEqual(result.state,'lost');self.assertIsNone(result.rotation);self.assertIsNone(result.translation)
        self.assertIsNone(tracker.pose)

    def test_independent_foreground_removes_occluder_correspondences_before_pose(self):
        _,points,k,pose,tracker=fixture();tracker.replenish=False
        pixels=projected(points,*pose,k);mask=np.full(tracker.previous_gray.shape,255,np.uint8)
        for x,y in np.round(pixels[:8]).astype(int):mask[y-2:y+3,x-2:x+3]=0
        tracker.external_foreground=lambda gray,frame:(mask,dict(maskReason=None))
        tracker.match=lambda gray:(np.empty((0,2)),np.empty((0,3)))
        with patch('bench.surface_tracker.follow',return_value=(pixels,np.ones(25,bool))):
            result=tracker.update(tracker.previous_gray,k,1)
        self.assertEqual(result.state,'tracking')
        self.assertEqual(result.statistics['featuresExcludedByMask'],8)
        self.assertEqual(result.statistics['correspondences'],17)

    def test_missing_foreground_hides_design_while_preserving_valid_private_pose(self):
        _,points,k,pose,tracker=fixture();tracker.replenish=False
        tracker.external_foreground=lambda gray,frame:(None,dict(maskReason='prompt_motion_unreliable'))
        tracker.match=lambda gray:(np.empty((0,2)),np.empty((0,3)))
        with patch('bench.surface_tracker.follow',return_value=(projected(points,*pose,k),np.ones(25,bool))):
            result=tracker.update(tracker.previous_gray,k,1)
        self.assertEqual(result.state,'limited');self.assertEqual(result.reason,'foreground_unavailable')
        self.assertIsNone(result.rotation);self.assertIsNotNone(tracker.pose)


if __name__=='__main__':unittest.main()
