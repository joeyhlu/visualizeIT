"""Independent visibility checks for RGB-only HD evaluation."""
import unittest
import numpy as np
from .hd_experiment import unoccluded, annotations, tracking_input
from .tracker import TrackingResult
from .fast_tracker import FastRigidTracker
from .vision import cv2
from .speed_sweep import rescore_visible_rows


class HDTests(unittest.TestCase):
    def test_shared_visibility_scoring_replaces_observations_and_preserves_lost_frames(self):
        truth=[dict(frame=10,point=0,valid=0,expected_x=4.,expected_y=4.,observed_x='',observed_y='')]
        points=np.array([[0,0,1.]])
        calibration=dict(cam_K=[10,0,4,0,10,4,0,0,1])
        tracked=TrackingResult(10,'tracking',None,np.eye(3),np.array([.1,0,0]),{})
        actual=rescore_visible_rows(truth,points,tracked,calibration,720)
        self.assertEqual(actual[0]['observed_x'],5.)
        self.assertEqual(actual[0]['expected_x'],4.)
        self.assertEqual(actual[0]['valid'],1)
        lost=TrackingResult(10,'lost','test',None,None,{})
        failure=rescore_visible_rows(actual,points,lost,calibration,720)
        self.assertEqual(failure[0]['valid'],0)
        self.assertEqual(failure[0]['observed_x'],'')
        self.assertEqual(len(failure),1)

    def test_object_region_expands_and_recovers_global_search_after_loss(self):
        gray=np.random.default_rng(13).integers(0,256,(360,480),dtype=np.uint8)
        features,descriptors=cv2.ORB_create(nfeatures=400).detectAndCompute(gray,None)
        points=np.random.default_rng(14).normal(size=(len(features),3))
        pixels=np.tile([[200.,150.],[260.,210.]],(6,1))
        tracker=FastRigidTracker(points,descriptors,points,gray,pixels,points[:12],use_roi=True)
        initial=tracker.search_region(gray.shape)
        self.assertLess(initial[0],pixels[:,0].min());self.assertLess(initial[1],pixels[:,1].min())
        self.assertGreater(initial[2],pixels[:,0].max());self.assertGreater(initial[3],pixels[:,1].max())
        tracker.search_failures=2
        expanded=tracker.search_region(gray.shape)
        self.assertLess(expanded[0],initial[0]);self.assertGreater(expanded[2],initial[2])
        tracker.search_failures=3
        self.assertEqual(tracker.search_region(gray.shape),(0,0,480,360))

    def test_object_region_matching_returns_full_camera_image_coordinates(self):
        gray=np.random.default_rng(21).integers(0,256,(360,480),dtype=np.uint8)
        pixels=np.tile([[180.,130.],[280.,230.]],(6,1))
        features,descriptors=cv2.ORB_create(nfeatures=800).detectAndCompute(gray[80:281,130:331],None)
        points=np.random.default_rng(22).normal(size=(len(features),3))
        tracker=FastRigidTracker(points,descriptors,points,gray,pixels,points[:12],use_roi=True)
        observed,matched=tracker.match(gray)
        self.assertGreater(len(matched),len(points)*.8)
        for pixel,point in zip(observed,matched):
            index=np.flatnonzero(np.all(points==point,axis=1))[0]
            np.testing.assert_allclose(pixel,np.array(features[index].pt)+[130,80])

    def test_resize_first_has_same_geometry_and_bounded_luminance_difference(self):
        rgb=np.random.default_rng(11).integers(0,256,(144,192,3),dtype=np.uint8)
        calibration=dict(cam_K=[100,0,96,0,100,72,0,0,1])
        before,k=tracking_input(rgb,calibration,36,True)
        after,other_k=tracking_input(rgb,calibration,36,True,True)
        np.testing.assert_array_equal(k,other_k)
        self.assertEqual(after.shape,(36,48))
        self.assertLessEqual(np.abs(before.astype(int)-after.astype(int)).max(),1)
        unchanged,_=tracking_input(rgb,calibration,144,True,True)
        np.testing.assert_array_equal(unchanged,cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))

    def test_indexed_binary_matching_recovers_known_reference_associations(self):
        gray=np.random.default_rng(9).integers(0,256,(240,320),dtype=np.uint8)
        orb=cv2.ORB_create(nfeatures=800);features,descriptors=orb.detectAndCompute(gray,None)
        points=np.random.default_rng(10).normal(size=(len(features),3))
        for key in (12,16):
            with self.subTest(key_size=key):
                tracker=FastRigidTracker(points,descriptors,points,index_key_size=key)
                pixels,matched=tracker.match(gray)
                self.assertGreater(len(matched),len(points)*.8)
                for pixel,point in zip(pixels,matched):
                    index=np.flatnonzero(np.all(points==point,axis=1))[0]
                    np.testing.assert_allclose(pixel,features[index].pt)
                self.assertEqual(len(np.unique(matched,axis=0)),len(matched))

    def test_resizing_preserves_camera_projection_with_pixel_center_mapping(self):
        rgb=np.zeros((12,16,3),dtype=np.uint8)
        k=np.array([[8,0,7.5],[0,8,5.5],[0,0,1.]])
        image,scaled=tracking_input(rgb,dict(cam_K=k.ravel().tolist()),6)
        point=np.array([.2,.3,1]);native=(k@point)[:2]
        np.testing.assert_allclose((scaled@point)[:2],(native+.5)*.5-.5)
        self.assertEqual(image.shape,(6,8,3))

    def test_ray_visibility_rejects_surface_hidden_behind_another_triangle(self):
        vertices=np.array([[-1,-1,1],[1,-1,1],[0,1,1]],dtype=float)
        points=np.array([[0,0,1],[0,0,2],[3,0,2]],dtype=float)
        np.testing.assert_array_equal(unoccluded(points,vertices,np.array([[0,1,2]]),np.zeros(3)),[True,False,True])

    def test_lost_pose_still_counts_visible_evaluation_landmarks(self):
        record=dict(frameId=10,evaluationPose=dict(cam_R_m2c=np.eye(3).ravel().tolist(),cam_t_m2c=[0,0,1000]),
                    calibration=dict(cam_K=[10,0,4,0,10,4,0,0,1]))
        points=np.array([[0,0,0.]])
        vertices=np.array([[-.1,-.1,0],[.1,-.1,0],[0,.1,0]])
        estimate=TrackingResult(10,'lost','test',None,None,{})
        rows=annotations(record,points,np.array([[0,0,-1.]]),vertices,np.array([[0,1,2]]),
                         np.ones((8,8),dtype=bool),estimate)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['valid'],0)
        self.assertEqual(rows[0]['observed_x'],'')


if __name__=='__main__': unittest.main()
