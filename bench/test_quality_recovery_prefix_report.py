import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from .quality_recovery_prefix_report import compare


class RecoveryPrefixTests(unittest.TestCase):
    def test_provenance_and_native_mask_contents_are_required(self):
        with TemporaryDirectory() as directory:
            a=Path(directory)/'a'; b=Path(directory)/'b'; a.mkdir(); b.mkdir()
            for p in (a,b):
                (p/'0.png').write_bytes(b'setup'); (p/'1.png').write_bytes(b'foreground')
            pose=dict(frameId=1,mask_state='available',pose_state='tracking',render_state='visible',
                failure_reason=None,mask_sha256='known',cameraFromObject=None)
            result=dict(frames=[pose],provenance={'model':'pinned'},tracking_settings={},stress_test={})
            masks=dict(frames=[dict(frameId=i,path=f'{i}.png',mask_state='available') for i in (0,1)],
                provenance=result['provenance'],stress_test={},mask_association=True)
            self.assertTrue(compare(result,result,masks,masks,a,b)['prefix_passed'])
            pose_stage=copy.deepcopy(result);pose_stage['provenance']['foundpose_bank_sha256']='pose-only-bank'
            self.assertTrue(compare(pose_stage,pose_stage,masks,masks,a,b)['prefix_passed'])
            changed_bank=copy.deepcopy(pose_stage);changed_bank['provenance']['foundpose_bank_sha256']='different-bank'
            self.assertFalse(compare(pose_stage,changed_bank,masks,masks,a,b)['prefix_passed'])
            changed=copy.deepcopy(result); changed['provenance']['model']='different'
            self.assertFalse(compare(result,changed,masks,masks,a,b)['prefix_passed'])
            (b/'1.png').write_bytes(b'different foreground')
            report=compare(result,result,masks,masks,a,b)
            self.assertFalse(report['prefix_passed']); self.assertEqual(report['mask_differing_source_frames'],[1])

    def test_missing_source_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Ordered source prefix'):
            compare(dict(frames=[dict(frameId=3)]),dict(frames=[dict(frameId=4)]),{}, {},Path('.'),Path('.'))


if __name__=='__main__':unittest.main()
