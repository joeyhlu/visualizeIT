from types import SimpleNamespace
import unittest
import numpy as np
from .quality_surface_agreement import sample_surface


class SurfaceSamplingTests(unittest.TestCase):
    def test_sampling_follows_area_and_stays_inside_triangles(self):
        mesh=SimpleNamespace(positions=np.array([[0.,0,0],[2,0,0],[0,1,0],[5,0,0],[11,0,0],[5,3,0]]),
            normals=np.tile([0.,0,1],(6,1)),triangles=np.array([[0,1,2],[3,4,5]]))
        points,normals,faces,bary=sample_surface(mesh,1000)
        self.assertEqual((faces==1).sum(),900)
        self.assertTrue((bary>=0).all()); np.testing.assert_allclose(bary.sum(axis=1),1)
        np.testing.assert_allclose(points,np.einsum('ni,nij->nj',bary,mesh.positions[mesh.triangles[faces]]))
        np.testing.assert_allclose(normals,np.tile([0,0,1],(1000,1)))
        np.testing.assert_array_equal(points,sample_surface(mesh,1000)[0])

    def test_degenerate_surface_cannot_silently_be_scored(self):
        mesh=SimpleNamespace(positions=np.zeros((3,3)),normals=np.zeros((3,3)),triangles=np.array([[0,1,2]]))
        with self.assertRaisesRegex(ValueError,'Nondegenerate'):sample_surface(mesh)


if __name__=='__main__':unittest.main()
