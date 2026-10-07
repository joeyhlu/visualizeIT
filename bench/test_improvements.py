import unittest
import numpy as np
from .model import Mesh
from .mapping_candidates import correct_scale, align_charts, has_overlaps, design_uv
from .ply_model import pose_camera
from .tracker import RigidTracker
from .vision import cv2
from .renderer import render
from .model import Camera
from .video_experiment import composite, annotated_pose


class ImprovementTests(unittest.TestCase):
    def test_pattern_scale_aspect_and_rotation(self):
        np.testing.assert_allclose(design_uv(np.array([[.04,.08]]),True,tile=.04,aspect=2),[[1,1]])
        np.testing.assert_allclose(design_uv(np.array([[.04,0]]),True,tile=.04,rotation_degrees=90),[[0,1]],atol=1e-12)
    def test_uniform_chart_scale_corrects_two_to_one_map(self):
        mesh = Mesh('plane', [[0,0,0],[1,0,0],[0,1,0]], [[0,0,1]]*3, [[0,0],[.5,0],[0,.5]], [[0,1,2]])
        uv = correct_scale(mesh, mesh.uv, np.zeros(3, dtype=int))
        np.testing.assert_allclose(uv[1]-uv[0], [1,0])
        np.testing.assert_allclose(uv[2]-uv[0], [0,1])

    def test_overlap_detects_area_but_not_shared_boundary(self):
        uv = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
        self.assertFalse(has_overlaps(uv, np.array([[0,1,2],[0,2,3]])))
        self.assertTrue(has_overlaps(np.concatenate([uv, uv+.1]), np.array([[0,1,2],[4,5,6]])))

    def test_seam_alignment_handles_rotated_translated_charts(self):
        positions = np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,0],[1,1,0],[0,1,0]], dtype=float)
        uv = positions[:,:2].copy()
        uv[3:] = uv[3:]@np.array([[0,-1],[1,0]])+np.array([4,7])
        mesh = Mesh('two charts', positions, [[0,0,1]]*6, uv, [[0,1,2],[3,4,5]])
        aligned = align_charts(mesh, np.array([[0,1,2],[0,2,3]]), np.array([0,0,0,1,1,1]), uv, np.ones(2), False)
        np.testing.assert_allclose(aligned[0], aligned[3], atol=1e-8)
        np.testing.assert_allclose(aligned[2], aligned[4], atol=1e-8)

    def test_bop_pose_projection_matches_direct_source_projection(self):
        rotation = cv2.Rodrigues(np.array([.2,.1,-.3]))[0]
        translation = np.array([.02,-.01,.7])
        transform = np.eye(4); transform[:3,:3] = [[1,0,0],[0,0,1],[0,1,0]]; transform[:3,3] = [.1,.2,-.05]
        k = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=float)
        camera = pose_camera(rotation, translation, {'cam_K':k.ravel().tolist()}, transform,640,480)
        points = np.array([[.01,.02,.03],[-.02,.01,-.01]])
        bench = points@transform[:3,:3].T+transform[:3,3]
        projected,_ = camera.project(bench)
        reference = cv2.projectPoints(points, cv2.Rodrigues(rotation)[0], translation, k,None)[0].reshape(-1,2)
        np.testing.assert_allclose(projected,reference,atol=1e-10)

    def test_pnp_recovers_pose_with_noise_and_outliers(self):
        rng = np.random.default_rng(17)
        points = rng.uniform(-.05,.05,(80,3))
        rotation = np.array([.15,-.2,.05]); translation = np.array([.01,-.02,.6])
        k = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=float)
        pixels = cv2.projectPoints(points,rotation,translation,k,None)[0].reshape(-1,2)
        pixels += rng.normal(0,.2,pixels.shape); pixels[:10] += 40
        tracker = RigidTracker(points,np.zeros((80,32),np.uint8),points)
        result = tracker.solve(pixels,points,k,1)
        self.assertIsInstance(result,tuple)
        result = result[0]
        self.assertEqual(result.state,'tracking')
        np.testing.assert_allclose(result.translation,translation,atol=.001)
        self.assertGreater(result.statistics['inliers'],60)

    def test_ambiguous_and_missing_pose_never_returns_visible_transform(self):
        points = np.zeros((20,3)); points[:,:2] = np.random.default_rng(4).normal(0,.02,(20,2))
        tracker = RigidTracker(points,np.zeros((20,32),np.uint8),points)
        k = np.eye(3)
        result = tracker.solve(points[:,:2],points,k,1)
        self.assertEqual(result.state,'limited'); self.assertIsNone(result.rotation)
        result = tracker.update(np.zeros((100,100,3),np.uint8),k,2)
        self.assertEqual(result.state,'lost'); self.assertIsNone(result.translation)

    def test_repeating_image_keeps_unwrapped_surface_coordinates(self):
        mesh = Mesh('texture', [[-.3,-.3,1],[.3,-.3,1],[-.3,.3,1]],[[0,0,-1]]*3,[[1.25,.25]]*3,[[0,2,1]])
        camera = Camera.look_at([0,0,0],[0,0,1],width=48,height=48,focal=40)
        image = np.array([[[200,0,0],[0,200,0]],[[0,0,200],[200,200,200]]],dtype=np.uint8)
        wrapped = render(mesh,camera,texture_image=image,texture_repeat=True)
        mesh.uv[:]=[.25,.25]
        reference=render(mesh,camera,texture_image=image)
        np.testing.assert_allclose(wrapped.color,reference.color)
        np.testing.assert_allclose(wrapped.uv[26,18],[1.25,.25])

    def test_measured_depth_occlusion_and_unknown_depth(self):
        mesh=Mesh('occluded',[[-.3,-.3,1],[.3,-.3,1],[-.3,.3,1]],[[0,0,-1]]*3,[[0,0]]*3,[[0,2,1]])
        camera=Camera.look_at([0,0,0],[0,0,1],width=48,height=48,focal=40)
        frame=render(mesh,camera)
        rgb=np.zeros((48,48,3),dtype=np.uint8)
        for depth in (np.zeros((48,48)),np.full((48,48),.5)):
            output,mask=composite(rgb,depth,frame)
            self.assertFalse(mask.any()); np.testing.assert_array_equal(output,rgb)
        _,mask=composite(rgb,np.full((48,48),1.),frame)
        self.assertTrue(mask.any())

    def test_rounded_annotation_rotation_repaired_and_reflection_rejected(self):
        rotation=cv2.Rodrigues(np.array([.2,.1,.3]))[0]
        record={'evaluationPose':{'cam_R_m2c':np.round(rotation,6).ravel().tolist(),'cam_t_m2c':[0,0,500]}}
        repaired,translation=annotated_pose(record)
        np.testing.assert_allclose(repaired@repaired.T,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(translation,[0,0,.5])
        record['evaluationPose']['cam_R_m2c']=np.diag([1,1,-1]).ravel().tolist()
        with self.assertRaises(ValueError): annotated_pose(record)


if __name__ == '__main__': unittest.main()
