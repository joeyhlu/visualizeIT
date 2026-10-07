import unittest
import numpy as np
from .quality_photometric import bilinear, align_samples, PhotometricSettings, project_samples


class PhotometricTests(unittest.TestCase):
    def test_double_precision_sampling_preserves_subpixel_motion(self):
        image = np.tile(np.arange(30, dtype=float), (30, 1))
        result, valid = bilinear(image, [[12.000001, 8.2], [12.000002, 8.2], [-1, 2]])
        self.assertAlmostEqual(result[1]-result[0], .000001, places=10)
        self.assertEqual(valid.tolist(), [True, True, False])

    def test_flat_texture_cannot_propose_a_pose(self):
        pose = np.eye(4); pose[2, 3] = .5
        candidate, stats = align_samples(np.full((40, 40, 3), 100, np.uint8), np.ones((40, 40)),
            np.eye(3), pose, np.zeros((250, 3)), np.full(250, .4))
        np.testing.assert_array_equal(candidate, pose)
        self.assertEqual(stats['state'], 'unobservable_texture')

    def test_recovers_small_displacement_on_a_known_curved_surface(self):
        # Analytical textured height field supplies truth independently of the optimizer.
        height = width = 160; k = np.array([[170., 0, 80], [0, 170., 80], [0, 0, 1]])
        yy, xx = np.mgrid[:height, :width]
        texture = .45+.13*np.sin(xx*.18)+.12*np.cos(yy*.15)+.1*np.sin((xx+yy)*.12)
        rgb = np.repeat(np.rint(texture*255).astype(np.uint8)[..., None], 3, 2)
        x, y = np.meshgrid(np.arange(35, 125, 3), np.arange(35, 125, 3)); x=x.ravel(); y=y.ravel()
        z = .5+.06*np.sin(x*.035)*np.cos(y*.029)
        xyz = np.column_stack(((x+.5-80)/170*z, (y+.5-80)/170*z, z))
        truth = np.eye(4); truth[2, 3] = .5; points = xyz-truth[:3, 3]
        seed = truth.copy(); seed[:3, 3] += [.003, -.002, .001]
        pose, stats = align_samples(rgb, np.ones((height, width)), k, seed, points, texture[y,x],
            PhotometricSettings(maximum_evaluations=80))
        before = np.linalg.norm(project_samples(points, seed, k)[0]-np.column_stack((x,y)), axis=1).mean()
        after = np.linalg.norm(project_samples(points, pose, k)[0]-np.column_stack((x,y)), axis=1).mean()
        self.assertEqual(stats['state'], 'proposal')
        self.assertLess(after, before*.25)
        self.assertFalse(stats['tracking_validated'])


if __name__ == '__main__': unittest.main()
