import unittest
from .quality_recovery_report import strict_recovery
from . import test_quality as fixture


class RecoveryReportTests(unittest.TestCase):
    def run_synthetic(self):
        tracker=fixture.SequentialTracker(fixture.Backend(),fixture.POSE); frames=[]
        for i in range(60):
            mask=fixture.MaskPrediction(i,None,'lost','blank_current_image') if 10<=i<25 else fixture.segmentation(i)
            frames.append(tracker.update(fixture.frame(i),mask))
        return dict(frames=frames)

    def test_two_validated_frames_and_hidden_suppression_are_required(self):
        result=self.run_synthetic(); self.assertTrue(strict_recovery(result,10)['passed'])
        result['frames'][10]['cameraFromObject']=fixture.POSE.tolist()
        self.assertFalse(strict_recovery(result,10)['passed'])

    def test_fake_tracking_and_missing_frames_cannot_pass_recovery(self):
        result=self.run_synthetic(); result['frames'][26]['cameraFromObject']=None
        self.assertFalse(strict_recovery(result,10)['passed'])
        result=self.run_synthetic(); result['frames'].pop(12)
        with self.assertRaisesRegex(ValueError,'consecutive'):strict_recovery(result,10)


if __name__=='__main__':unittest.main()
