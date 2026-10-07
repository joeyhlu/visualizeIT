import unittest
from .quality_joint_seed import choose


def candidate(angle,shape,detail):
    return dict(degrees=angle,visible_mask_silhouette_iou=shape,visible_overlap=.9,detail=dict(detail_correlation=detail))


class JointEvidenceTests(unittest.TestCase):
    def test_outline_only_and_texture_only_gains_do_not_displace_the_seed(self):
        seed=candidate(0,.8,.6)
        selected,reason=choose([seed,candidate(90,.9,.55),candidate(180,.7,.9)])
        self.assertIs(selected,seed);self.assertEqual(reason,'no_joint_evidence')

    def test_small_shape_ties_and_uninformative_texture_do_not_choose_large_rotation(self):
        seed=candidate(0,.8,.6)
        self.assertIs(choose([seed,candidate(180,.802,.9)])[0],seed)
        flat=candidate(0,.8,None)
        self.assertEqual(choose([flat,candidate(180,.9,.9)])[1],'texture_uninformative')

    def test_joint_improvement_with_actual_foreground_support_can_propose_a_seed(self):
        seed=candidate(0,.8,.6);both=candidate(60,.85,.8)
        self.assertIs(choose([seed,both])[0],both)
        both['visible_overlap']=.2
        self.assertIs(choose([seed,both])[0],seed)
