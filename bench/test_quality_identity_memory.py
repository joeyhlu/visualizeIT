import unittest
import numpy as np
from .quality_contract import PoseCandidate
from .quality_identity_memory import SurfaceIdentityMemory
from . import test_quality_memory as fixtures


class IdentityMemoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ModelMemoryTests(); self.fixture.setUp()
        self.memory = SurfaceIdentityMemory(lambda pose, k, w, h: np.ones((h, w), np.float32))

    def candidate(self, pose):
        return PoseCandidate(pose, np.empty((0, 3)), np.empty((0, 2)), np.empty(0))

    def test_can_check_identity_after_blank_frame_without_flow(self):
        f = self.fixture
        self.assertTrue(self.memory.commit(f.frame(0), f.mask, f.pose))
        self.assertEqual(self.memory.check(f.frame(1, blank=True), np.zeros_like(f.mask), self.candidate(f.pose))[2]['state'], 'unavailable')
        valid, _, stats = self.memory.check(f.frame(2), f.mask, self.candidate(f.pose))
        self.assertEqual(stats['state'], 'available'); self.assertTrue(valid)
        flip = f.pose.copy(); flip[:3, :3] = np.diag([-1., -1., 1.])
        self.assertEqual(self.memory.check(f.frame(3), f.mask, self.candidate(flip))[1], 'contradicts_surface_identity')

    def test_textureless_image_cannot_become_identity_evidence(self):
        f = self.fixture
        self.assertFalse(self.memory.commit(f.frame(0, blank=True), f.mask, f.pose))
        self.assertEqual(self.memory.check(f.frame(1), f.mask, self.candidate(f.pose))[2]['state'], 'unavailable')


if __name__ == '__main__': unittest.main()
