import unittest
import numpy as np
from .visible_mask import VisibleMask


class VisibleMaskTests(unittest.TestCase):
    def test_contrasting_finger_is_excluded_from_an_initialized_object(self):
        gray=np.full((160,180),40,np.uint8);gray[35:135,40:140]=235
        gray[75:105,15:105]=95  # The occluder connects to background outside the object.
        geometry=np.zeros_like(gray);geometry[35:135,40:140]=255
        mask,statistics=VisibleMask().estimate(gray,geometry)
        self.assertIsNotNone(mask);self.assertIsNone(statistics['maskReason'])
        self.assertLess((mask[75:105,40:100]>0).mean(),.05)
        self.assertGreater((mask[40:70,45:135]>0).mean(),.95)
        self.assertEqual(mask[20,20],0)

    def test_indistinguishable_appearance_abstains_instead_of_claiming_a_mask(self):
        gray=np.full((100,100),128,np.uint8);geometry=np.zeros_like(gray);geometry[25:75,25:75]=255
        mask,statistics=VisibleMask().estimate(gray,geometry)
        self.assertIsNone(mask);self.assertEqual(statistics['maskReason'],'ambiguous_object_background_appearance')


if __name__=='__main__':unittest.main()
