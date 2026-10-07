"""Motion-evidence tests with textured calibrated planar surfaces, not accuracy claims."""
import unittest
import numpy as np
from .vision import cv2
from .quality_contract import Frame, PoseCandidate
from .quality_memory import ModelPointMemory


class ModelMemoryTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.k = np.array([[300., 0, 160], [0, 300., 120], [0, 0, 1.]])
        self.pose = np.eye(4); self.pose[2, 3] = 1.
        self.mask = np.zeros((240, 320), np.uint8); self.mask[25:215, 30:290] = 255
        rng = np.random.default_rng(7)
        texture = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0)
        self.rgb = np.repeat(texture[..., None], 3, axis=2)
        self.memory = ModelPointMemory(lambda pose, k, w, h: np.ones((h, w), np.float32))

    def frame(self, i, shift=0, blank=False):
        image = np.zeros_like(self.rgb) if blank else cv2.warpAffine(self.rgb, np.float32([[1, 0, shift], [0, 1, 0]]), (320, 240))
        return Frame(i, image, self.k)

    def test_known_model_motion_and_wrong_rotation(self):
        self.memory.commit(self.frame(0), self.mask, self.pose)
        self.memory.observe(self.frame(1, 3), self.mask)
        seed = self.memory.seed(1)
        self.assertIsNotNone(seed)
        self.assertAlmostEqual(seed[0, 3], .01, delta=.002)
        valid = PoseCandidate(seed, np.empty((0, 3)), np.empty((0, 2)), np.empty(0))
        self.assertTrue(self.memory.validate(valid, 1)[0])
        flip = seed.copy(); flip[:3, :3] = np.diag([-1., -1., 1.])
        bad = PoseCandidate(flip, valid.points_object_m, valid.pixels_image, valid.weights)
        self.assertEqual(self.memory.validate(bad, 1)[1], 'contradicts_observed_model_motion')

    def test_motion_evidence_does_not_survive_hidden_image(self):
        self.memory.commit(self.frame(0), self.mask, self.pose)
        self.memory.observe(self.frame(1, blank=True), np.zeros_like(self.mask))
        self.assertIsNone(self.memory.seed(1))
        self.memory.observe(self.frame(2, 3), self.mask)
        self.assertIsNone(self.memory.seed(2))

    def test_memory_expires_without_new_model_confirmation(self):
        self.memory.max_age = 2
        self.memory.commit(self.frame(0), self.mask, self.pose)
        for i in range(1, 4): self.memory.observe(self.frame(i, i), self.mask)
        self.assertIsNone(self.memory.seed(3))

    def test_future_suffix_cannot_change_earlier_motion(self):
        self.memory.commit(self.frame(0), self.mask, self.pose)
        self.memory.observe(self.frame(1, 3), self.mask)
        earlier = self.memory.seed(1)
        self.memory.observe(self.frame(2, 6), self.mask)
        other = ModelPointMemory(lambda pose, k, w, h: np.ones((h, w), np.float32))
        other.commit(self.frame(0), self.mask, self.pose)
        other.observe(self.frame(1, 3), self.mask)
        np.testing.assert_array_equal(earlier, other.seed(1))


if __name__ == '__main__': unittest.main()
