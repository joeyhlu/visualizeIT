import unittest
import numpy as np
from .quality_texture_detail import detail_metrics,axial_pose


class TextureDetailTests(unittest.TestCase):
    def test_shading_similarity_cannot_hide_different_surface_texture(self):
        rng=np.random.default_rng(12);detail=rng.normal(0,.045,(96,96))
        shading=np.tile(np.linspace(.2,.8,96),(96,1))
        observed=np.clip(shading+detail,0,1)
        image=np.repeat(np.rint(observed[...,None]*255).astype(np.uint8),3,axis=2)
        matching=np.repeat((observed*.7+.1)[...,None],3,axis=2).astype(np.float32)
        other=np.repeat(np.clip(shading+np.roll(detail,15,axis=0),0,1)[...,None],3,axis=2).astype(np.float32)
        mask=np.ones((96,96),bool)
        self.assertGreater(detail_metrics(image,matching,mask)['detail_correlation'],.99)
        self.assertLess(detail_metrics(image,other,mask)['detail_correlation'],.15)

    def test_flat_or_missing_surface_is_not_texture_evidence(self):
        image=np.full((80,80,3),120,np.uint8);color=image.astype(np.float32)/255
        self.assertIsNone(detail_metrics(image,color,np.ones((80,80),bool))['detail_correlation'])
        self.assertEqual(detail_metrics(image,color,np.zeros((80,80),bool))['state'],'insufficient_interior')

    def test_axial_hypothesis_preserves_object_center_and_metric_scale(self):
        seed=np.eye(4);seed[:3,3]=[.03,-.01,.6];center=np.array([.1,.03,-.02])
        pose=axial_pose(seed,[0,1,0],center,120)
        np.testing.assert_allclose(pose[:3,:3]@center+pose[:3,3],center+seed[:3,3])
        self.assertAlmostEqual(np.linalg.norm(pose[:3,:3]@np.array([0,.2,0])),.2)
