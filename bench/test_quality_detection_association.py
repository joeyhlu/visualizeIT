import unittest
import numpy as np
from .quality_detection_association import associate


class DetectionAssociationTests(unittest.TestCase):
    def fixture(self):
        prior=np.zeros((100,100),bool);prior[30:80,40:70]=True
        object_mask=prior.copy();chair=np.zeros_like(prior);chair[0:15,0:15]=True
        return prior,np.stack([chair,object_mask])

    def test_current_object_supported_by_memory_beats_unrelated_high_score(self):
        prior,masks=self.fixture();selected,stats=associate([.6,.35],masks,prior,16)
        self.assertEqual(selected,1);np.testing.assert_array_equal(masks[selected],prior)
        self.assertEqual(stats['ranked'][0]['prior_iou'],1.)

    def test_weak_appearance_cannot_be_rescued_by_geometric_overlap(self):
        prior,masks=self.fixture();selected,_=associate([.6,.2],masks,prior,16)
        self.assertIsNone(selected)

    def test_memory_expires_and_never_returns_a_prior_as_a_foreground_mask(self):
        prior,masks=self.fixture();selected,stats=associate([.6,.35],masks,prior,31)
        self.assertIsNone(selected);self.assertIn('expired',stats['reason'])
        selected,stats=associate([.6],masks[:1],prior,16)
        self.assertIsNone(selected);self.assertIn('no_supported',stats['reason'])

    def test_two_separate_similar_candidates_are_ambiguous(self):
        prior=np.ones((100,100),bool);masks=np.zeros((2,100,100),bool)
        masks[0,:50]=True;masks[1,50:]=True
        selected,stats=associate([.4,.4],masks,prior,2)
        self.assertIsNone(selected);self.assertIn('ambiguous',stats['reason'])


if __name__=='__main__':unittest.main()
